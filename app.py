import streamlit as st
import requests
from datetime import datetime
import os
import json
import urllib.parse
import re
import time
import trafilatura  
from groq import Groq
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from supabase import create_client, Client

# مكتبات الـ PDF والمعالجة العربية
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

# --- جلب وتأمين المفاتيح البرمجية ---
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip().strip('"').strip("'")
SERPER_API_KEY = os.environ.get('SERPER_API_KEY', '').strip().strip('"').strip("'")
SUPABASE_URL = os.environ.get('SUPABASE_URL', '').strip().strip('"').strip("'")
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '').strip().strip('"').strip("'")

# جلب كود الترقية بأمان من البيئة السحابية المخفية
VALID_PRO_KEY = os.environ.get('VALID_PRO_KEY', 'HETAT_PRO_DEFAULT_2026').strip()

def apply_commercial_flat_design():
    st.markdown(
        """
        <style>
        /* إعدادات الاتجاهات والنصوص العربية */
        .stApp { direction: rtl; text-align: right; }
        div[data-baseweb="textarea"] textarea { direction: rtl !important; text-align: right !important; }
        div[data-baseweb="input"] input { direction: rtl !important; text-align: right !important; }
        .stMarkdown div p { direction: rtl; text-align: right; font-size: 1.05rem; }
        h1, h2, h3, h4, h5, h6 { text-align: right !important; direction: rtl !important; font-family: 'Segoe UI', Arial, sans-serif; }
        
        /* تصميم أزرار Flat 2.0 */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            background-color: #4A90E2;
            color: white;
            font-weight: bold;
            box-shadow: none;
        }
        .stButton>button:hover { background-color: #357ABD; }
        
        /* منع النصوص في الشريط الجانبي من الالتفاف رأسياً عند تضييق المساحة */
        section[data-testid="stSidebar"] * {
            white-space: nowrap !important;
        }
        
        /* إخفاء تام لأي محتوى هارب من السايدبار عندما يكون مغلقاً */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

TRUSTED_DOMAINS = [
    "bbc.com", "reuters.com", "cnn.com", "skynewsarabia.com", 
    "aljazeera.net", "france24.com", "asharq.com", "alarabiya.net",
    "dw.com", "un.org", "who.int", "youm7.com", "extranews.tv",
    "mena.org.eg", "www.wam.ae", "spa.gov.sa", "cairo24.com",
    "alqaheranews.net", "almasryalyoum.com", "shorouknews.com"
]

@st.cache_resource
def init_services():
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    try: embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    except: embed_model = None
    supabase_client = None
    if SUPABASE_URL and SUPABASE_KEY:
        try: supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except: pass
    return groq_client, embed_model, supabase_client

groq_client, embed_model, supabase_client = init_services()

def save_check_to_database(fact, verdict, final_answer):
    if supabase_client and fact and verdict:
        try: supabase_client.table("fact_checks").insert({"fact": fact, "verdict": verdict, "final_answer": final_answer}).execute()
        except: pass

def get_recent_checks(limit=5):
    if supabase_client:
        try:
            response = supabase_client.table("fact_checks").select("*").order("created_at", desc=True).limit(limit).execute()
            return response.data if response.data else []
        except: return []
    return []

def cosine_similarity(a, b):
    if a is None or b is None: return 0.0
    return dot(a, b) / (norm(a) * norm(b) + 1e-8)

def filter_and_rank_sources(fact, snippets, top_k=4):
    if not snippets: return []
    if not embed_model or not fact: return snippets[:top_k]
    try:
        vectors = embed_model.encode([s['text'] for s in snippets])
        fact_vector = embed_model.encode([fact])[0]
        for i, snippet in enumerate(snippets):
            base_conf = float(cosine_similarity(fact_vector, vectors[i]))
            source_url = snippet.get('source', '').lower()
            if any(trusted in source_url for trusted in TRUSTED_DOMAINS):
                base_conf += 0.25 
            snippet['confidence'] = base_conf
        return sorted(snippets, key=lambda x: x.get('confidence', 0), reverse=True)[:top_k]
    except: return snippets[:top_k]

def search_trusted_sources_sources_serper(query, api_key, num_results=4):
    if not api_key or not query: return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    data = {"q": query, "num": num_results}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        results = response.json()
        snippets = []
        for item in results.get("organic", []):
            snippet_text = item.get("snippet")
            link = item.get("link")
            title = item.get("title", "رابط مصدر")
            if snippet_text: snippets.append({"text": snippet_text, "source": link, "title": title})
        return snippets
    except: return []

def generate_optimized_search_queries(fact):
    model = get_active_model()
    if not model or not groq_client or not fact: return [fact]
    prompt = f"""أنت محقق صحفي رقمي متمكن. نريد البحث في جوجل للتحقق من هذا الادعاء بدقة: "{fact}".
قم بتوليد 3 عبارات بحث مختلفة تماماً وقوية تشمل مصطلحات صحفية بديلة وعبارة إنجليزية.
أعد الإجابة كـ قائمة JSON فقط وصارمة دون أي هوامش:
["استعلام 1", "استعلام 2", "English query"]"""
    try:
        response = groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        queries = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        return queries if isinstance(queries, list) else [fact]
    except: return [fact]

def extract_evidence_from_url(target_url, fact, top_sentences=3):
    if not target_url or not fact or not embed_model: return ""
    try:
        downloaded = trafilatura.fetch_url(target_url)
        if not downloaded: return ""
        full_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not full_text or not full_text.strip(): return ""
        sentences = re.split(r'[.\n।?!I،•●]', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        if not sentences: return ""
        fact_vector = embed_model.encode([fact])[0]
        sentence_vectors = embed_model.encode(sentences)
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = float(cosine_similarity(fact_vector, sentence_vectors[idx]))
            scored_sentences.append((score, sentence))
        top_evidences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:top_sentences]
        return " | ".join([ev[1] for ev in top_evidences])
    except: return ""

def get_current_live_date():
    months_ar = {1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو", 7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"}
    now = datetime.now()
    return f"{now.day} {months_ar[now.month]} {now.year}"

def get_active_model():
    if not groq_client: return None
    try:
        available_models = [m.id for m in groq_client.models.list().data]
        for preferred in ["qwen", "llama-3.3", "llama3-8b"]:
            match = next((m for m in available_models if preferred in m.lower() and "preview" not in m.lower()), None)
            if match: return match
        return available_models[0] if available_models else None
    except: return None

def extract_source_entity(fact):
    model = get_active_model()
    if not model or not groq_client or not fact: return None, None
    prompt = f'تحلل النص واستخرج منه أي جهة أو مؤسسة أو مسؤول نُسب إليها الكلام. أعد الإجابة بصيغة JSON فقط: {{"has_entity": true, "entity_name": "الاسم", "expected_domain": "who.int"}}. النص: "{fact}"'
    try:
        response = groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        data = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        if data.get("has_entity"): return data.get("entity_name"), data.get("expected_domain")
    except: pass
    return None, None

def parse_ai_response(full_text):
    if not full_text: return None, "خطأ في معالجة الرد الداخلي"
    if "<think>" in full_text and "</think>" in full_text:
        parts = full_text.split("</think>")
        return parts[0].replace("<think>", "").strip(), parts[1].strip()
    return None, full_text

def evaluate_fact_with_multi_tier(fact, tier1, tier2, tier3, entity_name):
    model = get_active_model()
    if not model: return "خطأ في الاتصال بنظام المحاكمة الذكي"
    
    def build_evidence_context(sources, label):
        if not sources: return "لا توجد مستندات كافية في هذا المستوى."
        context_parts = []
        for idx, s in enumerate(sources, 1):
            deep_evidence = extract_evidence_from_url(s['source'], fact)
            combined_text = f"موجز الخبر: {s['text']}"
            if deep_evidence: combined_text += f" | تفاصيل إضافية: {deep_evidence}"
            context_parts.append(f"🗒️ مستند ({idx}) [{label}] -\nرابط المصدر: {s['source']}\nالنص المعلوماتي: {combined_text}")
        return "\n\n".join(context_parts)

    c1 = build_evidence_context(tier1, "الموقع الرسمي")
    c2 = build_evidence_context(tier2, "الصحافة العالمية")
    c3 = build_evidence_context(tier3, "نتائج الويب الموسعة")
    
    prompt = f"""أنت رئيس تحرير محترف لغرفة أخبار. احكم على صحة الادعاء منطقياً بناءً على الأدلة المرفقة دون فذلكة.
الادعاء المراد فحصه: "{fact}"
الأدلة المستخرجة حياً:
{c1}\n{c2}\n{c3}

⚖️ القواعد:
- [VERDICT: TRUE] إذا أكدت التقارير وقوع الحدث أو وجود القوات/الطائرات أياً كان السبب.
- [VERDICT: FALSE] إذا نفت النصوص الحدث كلياً.
- [VERDICT: PARTIAL] إذا تحقق أصل الحدث بتفاصيل مغايرة.
- [VERDICT: INSUFFICIENT_EVIDENCE] إذا كانت النصوص صامتة تماماً عن موضوع الادعاء.

📥 صيغة الرد:
السطر الأول الوسم فقط مثل [VERDICT: TRUE]
السطر الثاني: 📌 الدليل الحاسم المستند عليه: واقتبس الجملة الحرفية.
السطر الثالث وما بعده: التحليل الصحفي المقتضب."""
    
    try:
        response = groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return response.choices[0].message.content.strip()
    except Exception as e: return f"خطأ أثناء معالجة الحكم: {e}"

# 🛠️ [تعديل دالة الـ PDF]: دعم طباعة المراجع والروابط بداخل التقرير لمنع المربعات وتقديم مستند موثق
def generate_arabic_pdf(fact, verdict, analysis, sources_list):
    pdf_filename = "Fact_Check_Report.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    story = []
    font_url = "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"
    font_path = "Amiri-Regular.ttf"
    try:
        if not os.path.exists(font_path):
            response = requests.get(font_url, timeout=10)
            with open(font_path, "wb") as f: f.write(response.content)
        pdfmetrics.registerFont(TTFont('ArabicFont', font_path))
        font_name = 'ArabicFont'
    except: font_name = 'Helvetica'

    styles = getSampleStyleSheet()
    def process_ar_text(text):
        if not text: return ""
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName=font_name, fontSize=18, leading=22, textColor=colors.HexColor('#4A90E2'), alignment=2)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontName=font_name, fontSize=11, leading=16, textColor=colors.HexColor('#222222'), alignment=2)
    verdict_style = ParagraphStyle('VerdictStyle', parent=styles['Normal'], fontName=font_name, fontSize=13, leading=18, textColor=colors.HexColor('#5CB85C'), alignment=2)
    link_style = ParagraphStyle('LinkStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#1B365D'), alignment=0) # روابط الإنجليزية يسار

    story.append(Paragraph(process_ar_text("🛡️ تقرير منصة المحقق الذكي لتدقيق الحقائق"), title_style))
    story.append(Spacer(1, 15))
    data = [
        [Paragraph(process_ar_text(fact), body_style), Paragraph(process_ar_text("الادعاء المراد فحصه"), body_style)],
        [Paragraph(process_ar_text(verdict), verdict_style), Paragraph(process_ar_text("الحكم والنتيجة"), body_style)]
    ]
    t = Table(data, colWidths=[350, 150])
    t.setStyle(TableStyle([('BACKGROUND', (1,0), (1,-1), colors.HexColor('#F5F5F5')), ('ALIGN', (0,0), (-1,-1), 'RIGHT'), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')), ('BOTTOMPADDING', (0,0), (-1,-1), 8), ('TOPPADDING', (0,0), (-1,-1), 8)]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # التحليل
    story.append(Paragraph(process_ar_text("📋 التفكيك والتحليل الاستقصائي:"), body_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(process_ar_text(analysis), body_style))
    story.append(Spacer(1, 15))
    
    # المراجع والروابط داخل الـ PDF
    if sources_list:
        story.append(Paragraph(process_ar_text("🔗 الروابط والمصادر الاستنادية (Sources):"), body_style))
        story.append(Spacer(1, 5))
        for idx, src in enumerate(sources_list, 1):
            src_text = f"[{idx}] {src['title']} "
            story.append(Paragraph(process_ar_text(src_text), body_style))
            story.append(Paragraph(src['source'], link_style))
            story.append(Spacer(1, 4))
            
    try:
        doc.build(story)
        return pdf_filename
    except: return None

# --- واجهة المستخدم الرئيسية ---
st.set_page_config(page_title="المُحقق الذكي - بث المصادر حياً", layout="centered")
apply_commercial_flat_design()

st.sidebar.title("🔑 نظام الاشتراكات ورخصة الاستخدام")
license_key = st.sidebar.text_input("أدخل مفتاح الترقية لنسخة Pro:", "", type="password")

if license_key == VALID_PRO_KEY:
    is_pro = True
    st.sidebar.success("👑 حسابك الآن: ترخيص احترافي (Pro Account)")
else:
    is_pro = False
    st.sidebar.info("👤 حسابك الحالي: الباقة المجانية")

if 'free_checks_count' not in st.session_state:
    st.session_state.free_checks_count = 0

st.header("🛡️ المُحقق الذكي")
st.caption(f"📅 تاريخ التدقيق الحالي: {get_current_live_date()}")

fact_to_check = st.text_area("أدخل المعلومة أو الخبر المراد فحصه:", "")

if st.button("بدء الفحص الجنائي الرقمي"):
    if not GROQ_API_KEY or not SERPER_API_KEY:
        st.error("🚨 خطأ في النظام: المفاتيح البرمجية (API Keys) غير متوفرة في بيئة التشغيل.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أو ادعاء أولاً.")
    elif not is_pro and st.session_state.free_checks_count >= 3:
        st.error("🚫 عذراً! لقد استهلكت حدك المجاني اليومي بالكامل (3 فحوصات).")
    else:
        status_container = st.empty()
        live_logs = []
        
        def update_log(message):
            live_logs.append(message)
            status_container.markdown("\n".join(live_logs))
            time.sleep(0.3)
            
        update_log("🧠 **جاري تشغيل المعالج اللغوي واستخراج الكيانات والجهات المذكورة...**")
        entity_name, expected_domain = extract_source_entity(fact_to_check)
        if entity_name:
            update_log(f"🎯 **تم تحديد جهة النسبة المستهدفة:** `{entity_name}` | النطاق المتوقع: `{expected_domain}`")
        else:
            update_log("🔍 **لم يتم رصد نسبة صريحة لمؤسسة؛ جاري الانتقال للويب العام مباشرة...**")

        if is_pro:
            update_log("⚙️ **[ميزة Pro]: جاري استخدام الـ Query Expansion لتوليد مصفوفة بحث متعددة اللغات...**")
            optimized_queries = generate_optimized_search_queries(fact_to_check)
        else:
            optimized_queries = [fact_to_check]
            update_log("🔍 **[باقة مجانية]: جاري تهيئة البحث باستخدام النص الحرفي للادعاء...**")

        # شبكات جمع المصادر والروابط المستند عليها
        tier1_sources, tier2_sources, tier3_sources = [], [], []
        all_discovered_sources = [] # 🌟 مصفوفة لتجميع كافة الروابط المستند عليها حياً
        
        if entity_name and expected_domain:
            update_log(f"🌐 **جاري مسح الأرشيف الرسمي لنطاق:** `{expected_domain}`...")
            tier1_sources = search_trusted_sources_sources_serper(f"site:{expected_domain} {fact_to_check}", SERPER_API_KEY, num_results=2)
            for s in tier1_sources:
                update_log(f"📌 **عثرت في المصادر الرسمية على مقال بعنوان:** _\"{s['title']}\"_*")
                all_discovered_sources.append(s)

        update_log("📰 **جاري مسح غرف الأخبار والوكالات العالمية الموثوقة (BBC, Reuters, Al Jazeera, etc.)...**")
        sites_query = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS[:6]])
        tier2_sources = search_trusted_sources_sources_serper(f"({sites_query}) {fact_to_check}", SERPER_API_KEY, num_results=3)
        for s in tier2_sources:
            update_log(f"🌐 **تم العثور على مستند في وكالة أنباء بعنوان:** _\"{s['title']}\"_*")
            all_discovered_sources.append(s)

        update_log("🚀 **جاري تفعيل الفحص الموسع وشبكة الفلترة الذكية للويب العام...**")
        raw_tier3 = []
        for q in optimized_queries:
            search_results = search_trusted_sources_sources_serper(q, SERPER_API_KEY, num_results=2 if is_pro else 1)
            raw_tier3.extend(search_results)
            
        seen_sources = set()
        unique_tier3 = []
        for item in raw_tier3:
            if item['source'] not in seen_sources:
                seen_sources.add(item['source'])
                unique_tier3.append(item)
                
        tier3_sources = filter_and_rank_sources(fact_to_check, unique_tier3, top_k=4 if is_pro else 2)
        for s in tier3_sources:
            update_log(f"🔗 **رصد نتيجة ويب وثيقة الصلة:** _\"{s['title']}\"_*")
            all_discovered_sources.append(s)

        update_log("🧠 **جاري الآن مقارنة المستندات والمطابقة الجنائية وصياغة الحكم النهائي...**")
        
        if not tier1_sources and not tier2_sources and not tier3_sources:
            status_container.empty()
            st.error("⚠️ [حكم المنصة]: غير كافي للحكم (INSUFFICIENT EVIDENCE)")
        else:
            evaluation_result = evaluate_fact_with_multi_tier(fact_to_check, tier1_sources, tier2_sources, tier3_sources, entity_name)
            thinking, final_answer = parse_ai_response(evaluation_result)
            status_container.empty()
            
            if thinking and is_pro:
                with st.expander("🧠 مذكرات التحليل الداخلي للمحقق (Chain of Thought) [Pro Only]:"):
                    st.write(thinking)
            
            st.subheader("⚖️ حكم منصة التحقق النهائي:")
            verdict_type = "خاطئ"
            clean_answer = final_answer
            
            if "[VERDICT: TRUE]" in final_answer:
                verdict_type = "صحيح"
                clean_answer = final_answer.replace("[VERDICT: TRUE]", "").strip()
                st.success(clean_answer)
            elif "[VERDICT: PARTIAL]" in final_answer:
                verdict_type = "جزئيًا صحيح"
                clean_answer = final_answer.replace("[VERDICT: PARTIAL]", "").strip()
                st.warning(clean_answer)
            elif "[VERDICT: FALSE]" in final_answer:
                verdict_type = "خاطئ"
                clean_answer = final_answer.replace("[VERDICT: FALSE]", "").strip()
                st.error(clean_answer)
            elif "[VERDICT: INSUFFICIENT_EVIDENCE]" in final_answer:
                verdict_type = "غير كافي للحكم"
                clean_answer = final_answer.replace("[VERDICT: INSUFFICIENT_EVIDENCE]", "").strip()
                st.info(clean_answer)
            else:
                st.info(final_answer)
            
            save_check_to_database(fact_to_check, verdict_type, clean_answer)
            
            # 🌟 [التحديث الجوهري]: استعراض المصادر والروابط المستند عليها للمستخدم في الواجهة مباشرة
            if all_discovered_sources:
                st.markdown("---")
                st.subheader("🔗 الروابط والمصادر التي استند عليها التحقيق:")
                # إزالة أي روابط مكررة ناتجة عن الفحص متعدد المستويات
                unique_final_sources = []
                seen_links = set()
                for src in all_discovered_sources:
                    if src['source'] not in seen_links:
                        seen_links.add(src['source'])
                        unique_final_sources.append(src)
                
                # عرض الروابط كـ أزرار أو نصوص تشعبية أنيقة
                for idx, src in enumerate(unique_final_sources, 1):
                    st.markdown(f"**[{idx}] [{src['title']}]({src['source']})**")
                    st.caption(f"المصدر الأصلي: {src['source']}")
            
            if is_pro:
                # تمرير الروابط لدالة الـ PDF ليتم كتابتها بداخل الملف المحمل أيضاً
                pdf_path = generate_arabic_pdf(fact_to_check, verdict_type, clean_answer, unique_final_sources)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 تحميل تقرير التحقق كملف PDF احترافي",
                            data=pdf_file,
                            file_name=f"Fact_Check_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
            else:
                st.session_state.free_checks_count += 1
                st.sidebar.metric(label="الفحوصات المجانية المستهلكة اليوم", value=f"{st.session_state.free_checks_count} / 3")
                st.markdown("---")
                st.info("💡 ميزة تحميل تقارير الـ PDF مخصصة لأعضاء الباقة الاحترافية (Pro).")
            
            st.markdown(" ")
            st.code(f"الادعاء: {fact_to_check}\nالحكم النهائي: {clean_answer}", language="text")

# --- قسم السجل العام ---
st.markdown("---")
st.subheader("🔔 آخر الشائعات التي تم تفكيكها حديثاً عبر المنصة:")
recent_items = get_recent_checks()
if recent_items:
    for item in recent_items:
        v_type = item.get('verdict', 'خاطئ')
        badge = "🔵" if "غير كافي" in v_type else ("🔴" if "خاطئ" in v_type else ("🟡" if "جزئي" in v_type else "🟢"))
        with st.expander(f"{badge} {item['fact'][:70]}..."):
            st.markdown(f"**الادعاء الأصلي:** {item['fact']}")
            st.markdown(f"**الحكم والتحليل:** {item['final_answer']}")
