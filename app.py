import streamlit as st
import requests
from datetime import datetime
import os
import json
from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

# جلب المفاتيح من بيئة التشغيل (Secrets)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
SERPER_API_KEY = os.environ.get('SERPER_API_KEY')

# تهيئة وتحميل النماذج لمرة واحدة وحفظها في الكاش لسرعة الأداء
@st.cache_resource
def load_models():
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        groq_client = None
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return groq_client, embed_model

groq_client, embed_model = load_models()

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-8)

def filter_top_snippets(fact, snippets, top_k=3):
    if not embed_model or not snippets:
        return snippets[:top_k]
    
    vectors = embed_model.encode([s['text'] for s in snippets])
    fact_vector = embed_model.encode([fact])[0]

    for i, snippet in enumerate(snippets):
        conf = float(cosine_similarity(fact_vector, vectors[i]))
        snippets[i]['confidence'] = conf

    return sorted(snippets, key=lambda x: x['confidence'], reverse=True)[:top_k]

def search_trusted_sources_serper(query, api_key, num_results=4):
    if not api_key:
        return []

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    data = {"q": query, "num": num_results}

    snippets = []
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        for item in results.get("organic", []):
            snippet_text = item.get("snippet")
            link = item.get("link")
            date_str = item.get("date", "")
            
            date_obj = None
            if date_str:
                try:
                    date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
                except:
                    date_obj = None
            
            if snippet_text:
                snippets.append({"text": snippet_text, "source": link, "date": date_obj})
        return snippets
    except:
        return []

def scrape_full_content(target_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(target_url, headers=headers, timeout=6)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
        full_text = " ".join([txt.get_text().strip() for txt in text_elements if txt.get_text().strip()])
        return full_text[:3000]
    except:
        return ""

def get_current_live_date():
    months_ar = {
        1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
        7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"
    }
    now = datetime.now()
    return f"{now.day} {months_ar[now.month]} {now.year}"

def get_active_model():
    """جلب نموذج مستقر من جروك ديناميكياً لتجنب الـ 404"""
    try:
        available_models = [m.id for m in groq_client.models.list().data]
        for preferred in ["qwen", "llama-3.3", "llama3-70b"]:
            match = next((m for m in available_models if preferred in m.lower() and "preview" not in m.lower()), None)
            if match:
                return match
        return available_models[0] if available_models else None
    except:
        return None

def extract_source_entity(fact):
    """
    دالة ذكاء اصطناعي في الخلفية تستخرج اسم المنظمة أو الموقع وتتوقع الدومين الخاص به
    """
    model = get_active_model()
    if not model or not groq_client:
        return None, None
        
    prompt = f"""
    حلل النص التالي واستخرج منه أي إشارة لصحيفة، موقع إخباري، منظمة، أو جهة رسمية نُسب إليها الكلام (مثال: منظمة الصحة العالمية، اليوم السابع، الفاو، نيويورك تايمز).
    أعد الإجابة بصيغة JSON فقط كالتالي دون أي نص إضافي قبل أو بعد الـ JSON:
    {{
      "has_entity": true أو false,
      "entity_name": "اسم الجهة المستخرجة بالعربية",
      "expected_domain": "الدومين التقريبي للموقع مثل who.int أو youm7.com أو un.org"
    }}
    
    النص: "{fact}"
    """
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        res_text = response.choices[0].message.content.strip()
        # تنظيف الجيسون لو احتوى على علامات برمجية
        res_text = res_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(res_text)
        if data.get("has_entity"):
            return data.get("entity_name"), data.get("expected_domain")
    except:
        pass
    return None, None

def parse_ai_response(full_text):
    if "<think>" in full_text and "</think>" in full_text:
        parts = full_text.split("</think>")
        return parts[0].replace("<think>", "").strip(), parts[1].strip()
    elif "</think>" in full_text:
        parts = full_text.split("</think>")
        return parts[0].strip(), parts[1].strip()
    return None, full_text

def evaluate_fact_with_ai(fact, general_snippets, entity_snippets, entity_name):
    model = get_active_model()
    if not model:
        return "خطأ: لم يتم العثور على نماذج نشطة في حساب جروك."

    # 1. تجميع المحتوى العام
    scraped_contexts = []
    for s in general_snippets:
        full_body = scrape_full_content(s['source'])
        scraped_contexts.append(f"[مصدر عام: {s['source']}]\nالمحتوى: {full_body if full_body else s['text']}")

    # 2. تجميع محتوى المنظمة الخاصة (إن وجدت)
    entity_contexts = []
    if entity_name and entity_snippets:
        for s in entity_snippets:
            full_body = scrape_full_content(s['source'])
            entity_contexts.append(f"[منشور رسمي داخل موقع {entity_name}: {s['source']}]\nالمحتوى: {full_body if full_body else s['text']}")

    context_general = "\n\n---\n\n".join(scraped_contexts)
    context_entity = "\n\n---\n\n".join(entity_contexts) if entity_contexts else "لم يتم العثور على نتائج داخل الموقع الرسمي لهذه الجهة."

    live_date = get_current_live_date()
    
    prompt = f"""
أنت خبير فحص حقائق صحفي محترف (Fact-Checking Agent).
🛑 [سياق زمني حرج]: تاريخ اليوم هو: {live_date}. محاكمة التواريخ تتم بناءً على هذا اليوم الحالي.

الادعاء المراد فحصها الآن: "{fact}"

المنظمة/الجهة المعنية بالتحقق المباشر (إن وجدت): {entity_name if entity_name else 'لا يوجد جهة محددة مسبقاً'}

🔍 [أولاً: البيانات المستخرجة حصرياً من داخل موقع الجهة المنسوب إليها الكلام]:
{context_entity}

🌐 [ثانياً: البيانات العامة من محركات البحث والوكالات الأخرى]:
{context_general}

⚙️ المنهجية المطلوبة:
1. فكك الادعاء.
2. إذا كان الادعاء ينسب كلاماً لجهة معينة (مثل "أعلنت منظمة كذا")، فتحقق أولاً: هل تؤكد وثائق هذه الجهة المرفقة أعلاه هذا الإعلان؟ وإذا لم تجده في موقعهم، وضح للمستخدم: "لا يوجد أثر لهذا البيان في المنصات الرسمية لـ [اسم الجهة]".
3. صغ الحكم بـ "صحيح"، "جزئيًا صحيح"، أو "خاطئ" مع ذكر السبب والبديل الحقيقي من التواريخ الحية الحالية.
"""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"حدث خطأ أثناء استدعاء نموذج جروك: {e}"

# --- واجهة مستخدم Streamlit ---
st.set_page_config(page_title="مُحقق الحقائق الذكي المطور", layout="centered")
st.header("🔍 نظام التأكد من الحقائق الذكي (الإصدار الصحفي المحترف)")
st.caption(f"📅 تاريخ النظام اللحظي: {get_current_live_date()}")

fact_to_check = st.text_area("أدخل المعلومة أو الخبر المراد فحصه بدقة:", "أعلنت منظمة الصحة العالمية عن انتهاء مرض السكري في 2026")

if st.button("بدء فحص وتفكيك الحقيقة"):
    if not GROQ_API_KEY or not SERPER_API_KEY:
        st.error("المفاتيح البرمجية ناقصة في إعدادات Secrets.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أولاً.")
    else:
        # 1. التقصي التلقائي عن وجود جهة أو منظمة منسوب إليها الخبر
        with st.spinner("جاري تحليل النص للكشف عن الجهات المنسوب إليها..."):
            entity_name, expected_domain = extract_source_entity(fact_to_check)
            
        # 2. بدء عمليات البحث المتوازي والمخصص
        with st.spinner("جاري كشط الويب وتجميع الأدلة والمقالات الكاملة..."):
            # أ. البحث العام في الويب
            current_year = datetime.now().year
            general_snippets = search_trusted_sources_serper(f"{fact_to_check} {current_year}", SERPER_API_KEY, num_results=4)
            
            # ب. البحث المخصص والضيق داخل موقع المنظمة الفعلي (لو وُجدت)
            entity_snippets = []
            if entity_name and expected_domain:
                st.toast(f"تم رصد نسبة الكلام لـ {entity_name}، جاري فحص أرشيف موقع {expected_domain}...", icon="🏢")
                specific_query = f"site:{expected_domain} {fact_to_check}"
                entity_snippets = search_trusted_sources_serper(specific_query, SERPER_API_KEY, num_results=3)

        if not general_snippets and not entity_snippets:
            st.warning("لم نتمكن من العثور على مصادر ويب حية متعلقة بهذا السياق حالياً.")
        else:
            # تصفية وتحديد أفضل المصادر العامة
            top_general = filter_top_snippets(fact_to_check, general_snippets, top_k=3)
            
            # عرض المصادر الذكية المكتشفة في الواجهة
            st.subheader("📌 المصادر والروابط الحية المكتشفة:")
            if entity_name and entity_snippets:
                st.markdown(f"**🔗 وثائق من داخل موقع ({entity_name}) الرسمي:**")
                for s in entity_snippets[:2]:
                    st.markdown(f"- [{s['source']}]({s['source']}) *(محتوى مخصص وموجه)*")
                    
            st.markdown("**🌐 مقالات وتغطيات صحفية عامة:**")
            for s in top_general:
                date_str = s['date'].strftime("%Y-%m-%d") if s['date'] else "تاريخ غير معلوم"
                st.markdown(f"- [{s['source']}]({s['source']}) *({date_str})*")
            
            # 3. صياغة وتقييم النتيجة بواسطة العميل الذكي
            with st.spinner("يقوم المحقق الآلي بمحاكمة البيانات ومطابقتها بأرشيف المنظمات..."):
                evaluation_result = evaluate_fact_with_ai(fact_to_check, top_general, entity_snippets, entity_name)
                
                # فصل التفكير عن النتيجة المعروضة
                thinking, final_answer = parse_ai_response(evaluation_result)
                
                if thinking:
                    with st.expander("🧠 رؤية مسار خطة المحقق وتقييم النسبة للمصدر الداخلي:"):
                        st.write(thinking)
                
                st.subheader("⚖️ حكم منصة التحقق:")
                if "صحيح" in final_answer and "جزئي" not in final_answer:
                    st.success(final_answer)
                elif "جزئي" in final_answer:
                    st.warning(final_answer)
                else:
                    st.error(final_answer)
