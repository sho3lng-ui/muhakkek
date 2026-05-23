import streamlit as st
import requests
from datetime import datetime
import os
import json
import urllib.parse
import re
import trafilatura  
from groq import Groq
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from supabase import create_client, Client

# --- جلب وتأمين المفاتيح البرمجية ---
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip().strip('"').strip("'")
SERPER_API_KEY = os.environ.get('SERPER_API_KEY', '').strip().strip('"').strip("'")
SUPABASE_URL = os.environ.get('SUPABASE_URL', '').strip().strip('"').strip("'")
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '').strip().strip('"').strip("'")

def apply_arabic_rtl():
    st.markdown(
        """
        <style>
        .stApp { direction: rtl; text-align: right; }
        div[data-baseweb="textarea"] textarea { direction: rtl !important; text-align: right !important; }
        div[data-baseweb="input"] input { direction: rtl !important; text-align: right !important; }
        .stMarkdown div p { direction: rtl; text-align: right; }
        .stAlert { direction: rtl; text-align: right; }
        h1, h2, h3, h4, h5, h6, p, span { text-align: right !important; direction: rtl !important; }
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
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    except:
        embed_model = None
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

# 🛠️ [تحديث]: تحسين الـ Ranking ليعطي أفضلية مطلقة للروابط التي تحتوي على كلمات مفتاحية موازية
def filter_and_rank_sources(fact, snippets, top_k=4):
    if not snippets: return []
    if not embed_model or not fact: return snippets[:top_k]
    try:
        vectors = embed_model.encode([s['text'] for s in snippets])
        fact_vector = embed_model.encode([fact])[0]
        for i, snippet in enumerate(snippets):
            base_conf = float(cosine_similarity(fact_vector, vectors[i]))
            source_url = snippet.get('source', '').lower()
            # دعم تفضيلي للمصادر الإخبارية الكبرى الموثوقة لمنع سقوطها في التصفية
            if any(trusted in source_url for trusted in TRUSTED_DOMAINS):
                base_conf += 0.25 
            snippet['confidence'] = base_conf
        return sorted(snippets, key=lambda x: x.get('confidence', 0), reverse=True)[:top_k]
    except:
        return snippets[:top_k]

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
            if snippet_text: snippets.append({"text": snippet_text, "source": link})
        return snippets
    except: return []

def generate_optimized_search_queries(fact):
    model = get_active_model()
    if not model or not groq_client or not fact: return [fact]
    prompt = f"""أنت محقق صحفي رقمي متمكن. نريد البحث في جوجل للتحقق من هذا الادعاء بدقة: "{fact}".
قم بتوليد 3 عبارات بحث مختلفة تماماً وقوية (شاملة الكلمات المفتاحية الأساسية، ومرادفات صحفية، وعبارة دقيقة باللغة الإنجليزية للوكالات العالمية مثل "Egypt Rafale UAE deployment").
أعد الإجابة كـ قائمة JSON فقط وصارمة دون أي هوامش:
["استعلام عربي 1", "استعلام عربي مرادف", "English search query"]"""
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
    prompt = f'تحلل النص واستخرج منه أي جهة نُسب إليها الكلام. أعد الإجابة بصيغة JSON فقط: {{"has_entity": true, "entity_name": "الاسم", "expected_domain": "who.int"}}. النص: "{fact}"'
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
    
    # 🛠️ [تحديث]: الـ Evidence Builder الهجين (يأخذ خلاصة النص من Serper + جمل المتن النظيف لضمان عدم ضياع السياق الإخباري)
    def build_evidence_context(sources, label):
        if not sources: return "لا توجد مستندات كافية في هذا المستوى."
        context_parts = []
        for idx, s in enumerate(sources, 1):
            deep_evidence = extract_evidence_from_url(s['source'], fact)
            # دمج خلاصة جوجل الفورية مع الجمل المستخلصة بعناية من المتن لضمان الكثافة المعلوماتية
            combined_text = f"موجز الخبر: {s['text']}"
            if deep_evidence:
                combined_text += f" | تفاصيل إضافية من داخل المقال: {deep_evidence}"
                
            context_parts.append(f"🗒️ مستند ({idx}) [{label}] -\nرابط المصدر: {s['source']}\nالنص المعلوماتي المتوفر: {combined_text}")
        return "\n\n".join(context_parts)

    c1 = build_evidence_context(tier1, "الموقع الرسمي الحكامي")
    c2 = build_evidence_context(tier2, "الصحافة العالمية الموثوقة")
    c3 = build_evidence_context(tier3, "نتائج الفحص الموسع للويب")
    
    prompt = f"""أنت رئيس تحرير محترف لغرفة أخبار ومنصة تدقيق حقائق عالمية. مهمتك هي الحكم على صحة الادعاء بناءً على المعنى المنطقي الواضح للأدلة المرفقة، دون تبريرات شخصية أو شروط تعجيزية.

الادعاء المراد فحصه: "{fact}"

إليك المستندات والنصوص المجلوبة حياً من الإنترنت حول هذا الادعاء:
[مستندات الصحافة والوكالات العالمية والمحلية المرفقة]:
{c1}
{c2}
{c3}

⚖️ [دليل قواعد الحكم الصحفي القطعي]:
- احكم بـ [VERDICT: TRUE] إذا كانت المستندات المرفقة (مثل تقارير BBC، سي إن إن، أو وكالات الأنباء) تذكر أو تؤكد وجود هذه الطائرات، أو القوات، أو مفرزة عسكرية، أو وقوع الحدث بالفعل في الواقع، بغض النظر عن سبب التواجد (سواء كان تدريباً، مناورة، أو انتشاراً استراتيجياً). التواجد الفعلي للطائرات أو القوات يثبت صحة أصل الادعاء.
- احكم بـ [VERDICT: FALSE] إذا كانت النصوص تنفي الحدث رسمياً أو تثبت عكسه كلياً.
- احكم بـ [VERDICT: PARTIAL] إذا تحقق أصل الحدث ولكن بتفاصيل أو أرقام مختلفة.
- احكم بـ [VERDICT: INSUFFICIENT_EVIDENCE] فقط وحصرياً إذا كانت النصوص المرفقة صامتة تماماً، أو قديمة جداً، ولا تمت بصلة نهائياً لموضوع الادعاء.

📥 [صيغة الرد الإلزامية]:
السطر الأول يحتوي فقط على أحد الأوسمة الأربعة: [VERDICT: TRUE] أو [VERDICT: FALSE] أو [VERDICT: PARTIAL] أو [VERDICT: INSUFFICIENT_EVIDENCE].
السطر الثاني: اكتب (📌 الدليل الحاسم المستند عليه): واقتبس الجملة الحرفية من المستندات المرفقة التي جعلتك تأخذ هذا القرار وثبتت الحادثة.
السطر الثالث وما بعده: اكتب التحليل الصحفي والتفكيك بأسلوب رصين ومختصر."""
    
    try:
        response = groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"خطأ أثناء معالجة الحكم الصحفي: {e}"

def display_share_buttons(fact, final_answer):
    st.markdown("---")
    st.markdown("📢 **شارك النتيجة لمنع انتشار الشائعات:**")
    share_text = f"🔍 تم التحقق من: '{fact}'\n⚖️ الحكم: {final_answer}"
    encoded_text = urllib.parse.quote(share_text)
    whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"
    twitter_url = f"https://twitter.com/intent/tweet?text={encoded_text}"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<a href="{whatsapp_url}" target="_blank"><button style="background-color:#25D366;color:white;border:none;padding:8px 12px;border-radius:5px;width:100%;cursor:pointer;font-weight:bold;">🟢 واتساب</button></a>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<a href="{twitter_url}" target="_blank"><button style="background-color:#1DA1F2;color:white;border:none;padding:8px 12px;border-radius:5px;width:100%;cursor:pointer;font-weight:bold;">🔵 إكس</button></a>', unsafe_allow_html=True)

# --- واجهة مستخدم Streamlit الرئيسية ---
st.set_page_config(page_title="(إصدار تجريبي) المُحقق الذكي", layout="centered")
apply_arabic_rtl()
st.header("🛡️ المُحقق الذكي للمعلومات والأخبار -- إصدار تجريبي")
st.caption(f"📅 تاريخ التحقق الحالي: {get_current_live_date()}")

fact_to_check = st.text_area("أدخل المعلومة أو الخبر المراد فحصه:", "")

if st.button("بدء الفحص الرقمي"):
    if not GROQ_API_KEY or not SERPER_API_KEY:
        st.error("🚨 خطأ في النظام: المفاتيح البرمجية غير متوفرة في بيئة التشغيل الحالية.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أو ادعاء أولاً ليقوم المحقق بفحصه.")
    else:
        tier1_sources, tier2_sources, tier3_sources = [], [], []
        
        with st.spinner("🕵️ جاري تحليل هندسة الادعاء وتوسيع نطاق البحث دلالياً باللغتين..."):
            optimized_queries = generate_optimized_search_queries(fact_to_check)
            st.caption(f"🔍 عوالم البحث النشطة الآن: {', '.join(optimized_queries)}")
        
        with st.spinner("🕵️ جاري سحب الأدلة الجنائية مصفوفياً وفحص السجلات الحية..."):
            entity_name, expected_domain = extract_source_entity(fact_to_check)
            
            if entity_name and expected_domain:
                tier1_sources = search_trusted_sources_sources_serper(f"site:{expected_domain} {fact_to_check}", SERPER_API_KEY, num_results=2)
            
            # 🛠️ [تحديث]: رفع عدد النتائج المسترجعة لضمان قنص وكالات الأنباء ومصادر التلفزيون بدقة
            sites_query = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS[:6]])
            tier2_sources = search_trusted_sources_sources_serper(f"({sites_query}) {fact_to_check}", SERPER_API_KEY, num_results=3)
            
            raw_tier3 = []
            for q in optimized_queries:
                search_results = search_trusted_sources_sources_serper(q, SERPER_API_KEY, num_results=3)
                raw_tier3.extend(search_results)
            
            seen_sources = set()
            unique_tier3 = []
            for item in raw_tier3:
                if item['source'] not in seen_sources:
                    seen_sources.add(item['source'])
                    unique_tier3.append(item)
            
            tier3_sources = filter_and_rank_sources(fact_to_check, unique_tier3, top_k=4)

        if not tier1_sources and not tier2_sources and not tier3_sources:
            st.error("⚠️ [حكم المنصة]: غير كافي للحكم (INSUFFICIENT EVIDENCE)")
            save_check_to_database(fact_to_check, "غير كافي للحكم", "لا توجد أدلة رقمية متوفرة على الويب حول هذا الادعاء.")
        else:
            evaluation_result = evaluate_fact_with_multi_tier(fact_to_check, tier1_sources, tier2_sources, tier3_sources, entity_name)
            thinking, final_answer = parse_ai_response(evaluation_result)
            
            if thinking:
                with st.expander("🧠 مذكرات التحليل الداخلي للمحقق (Chain of Thought):"):
                    st.write(thinking)
            
            st.subheader("⚖️ التحقق النهائي:")
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
                if "غير كاف" in final_answer or "غموض" in final_answer:
                    verdict_type = "غير كافي للحكم"
                    st.info(final_answer)
                elif "صحيح" in final_answer:
                    verdict_type = "صحيح"
                    st.success(final_answer)
                else:
                    st.error(final_answer)
            
            save_check_to_database(fact_to_check, verdict_type, clean_answer)
            display_share_buttons(fact_to_check, clean_answer)
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
else:
    st.info("لا توجد تدقيقات سابقة مسجلة في الأرشيف حتى الآن.")
