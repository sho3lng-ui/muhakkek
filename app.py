import streamlit as st
import requests
from datetime import datetime
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer

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

def search_trusted_sources_serper(fact, api_key, num_results=5, recent_year=2020):
    if not api_key:
        st.error("خطأ: مفتاح SERPER_API_KEY غير متوفر.")
        return []

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    data = {"q": fact, "num": num_results}

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
            
            if date_obj and date_obj.year < recent_year:
                continue
                
            if snippet_text:
                snippets.append({"text": snippet_text, "source": link, "date": date_obj})
        return snippets
    except Exception as e:
        st.error(f"خطأ أثناء جلب البيانات من Serper: {e}")
        return []

def scrape_full_content(target_url):
    """
    دالة آمنة ومستقرة لبيئة Streamlit لكشط وتنظيف محتوى الرابط بالكامل
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(target_url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # استخراج نصوص الفقرات والعناوين فقط لتخطي الـ Javascript والإعلانات والمناطق الميتة في الموقع
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
        full_text = " ".join([txt.get_text().strip() for txt in text_elements if txt.get_text().strip()])
        
        # قص النص حتى لا يتعدى سعة النموذج وسياق الـ API
        return full_text[:3500]
    except:
        return ""

def evaluate_fact_with_ai(fact, top_snippets):
    if not groq_client:
        return "خطأ: نموذج Groq غير مهيأ، تحقق من إضافة مفتاح GROQ_API_KEY بنجاح."
    if not top_snippets:
        return "لا توجد معلومات كافية لتقييم المعلومة."

    # كشط المحتويات الكاملة لأفضل المصادر التي تمت تصفيتها
    scraped_contexts = []
    for index, s in enumerate(top_snippets):
        with st.spinner(f"جاري قراءة وتحليل المصدر رقم {index+1} بالكامل..."):
            full_body = scrape_full_content(s['source'])
            content_to_use = full_body if full_body else s['text']
            scraped_contexts.append(f"[المصدر: {s['source']}]\nالمحتوى المتوفر: {content_to_use}")

    context = "\n\n---\n\n".join(scraped_contexts)
    
    prompt = f"""
أنت مساعد خبير، محايد، وصارم جداً مخصص للتحقق من الحقائق بمختلف اللغات (Fact-Checker).
⚠️ [قاعدة صارمة]: يمنع منعاً باتاً الاعتماد على مخزون معلوماتك الداخلي، مصادرك الوحيدة للحكم هي "المصادر المرفقة".

المعلومة المراد فحصها الآن: "{fact}"
المصادر المستخرجة:
{context}

ابدأ إجابتك بكلمة: "صحيح"، "جزئيًا صحيح"، أو "خاطئ"، ثم اذكر السبب المباشر في جملة واحدة إضافية.
"""

    try:
        # 1. جلب قائمة النماذج الفعالة حالياً من سيرفر جروك تلقائياً
        available_models = [m.id for m in groq_client.models.list().data]
        
        # 2. تحديد أولويات الاختيار (نبحث عن الأقوى في العربية والتحليل)
        selected_model = None
        for preferred in ["qwen-2.5", "llama-3.3", "llama3-70b", "mixstral"]:
            # البحث عن موديل يحتوي على هذا الاسم في القائمة الحية
            match = next((m for m in available_models if preferred in m.lower() and "preview" not in m.lower()), None)
            if match:
                selected_model = match
                break
        
        # إذا لم نجد أي من المفضلات، نأخذ أول نموذج متاح في السيرفر لضمان استمرار التطبيق
        if not selected_model and available_models:
            selected_model = available_models[0]
            
        if not selected_model:
            return "خطأ: لم يتم العثور على أي نماذج نشطة في حساب جروك الخاص بك."

        # طباعة اسم النموذج المستخدم في واجهة Streamlit كنوع من الشفافية والتأكيد
        st.caption(f"🤖 النموذج المستخدم حالياً للفحص: `{selected_model}`")

        # 3. استدعاء النموذج الذي تم اختياره ديناميكياً
        response = groq_client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"حدث خطأ أثناء استدعاء نموذج جروك: {e}"

# --- واجهة مستخدم Streamlit ---
st.set_page_config(page_title="مُحقق الحقائق الذكي", layout="centered")
st.header("🔍 نظام التأكد من الحقائق الذكي (النسخة المتقدمة)")

fact_to_check = st.text_area("أدخل المعلومة المراد فحصها:", "السرطان يقوي عظام المريض")

if st.button("بدء فحص الحقيقة"):
    if not GROQ_API_KEY or not SERPER_API_KEY:
        st.error("المفاتيح البرمجية ناقصة. تأكد من إضافة GROQ_API_KEY و SERPER_API_KEY في قسم Secrets وإعادة تشغيل التطبيق.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أو معلومة أولاً للفحص.")
    else:
        with st.spinner("جاري البحث عبر الويب وتجميع الروابط الحية..."):
            snippets = search_trusted_sources_serper(fact_to_check, SERPER_API_KEY)
            
            if not snippets:
                st.warning("لم نتمكن من العثور على مصادر ويب حديثة وموثوقة متعلقة بهذا النص.")
            else:
                top_snippets = filter_top_snippets(fact_to_check, snippets, top_k=3)
                
                st.subheader("📌 أهم المصادر الحية التي سيتم قراءتها بالكامل:")
                for s in top_snippets:
                    date_str = s['date'].strftime("%Y-%m-%d") if s['date'] else "تاريخ غير معلوم"
                    st.markdown(f"- [{s['source']}]({s['source']}) *({date_str})*")
                
                with st.spinner("جاري كشط محتوى المواقع وصياغة التقييم النهائي عبر الذكاء الاصطناعي..."):
                    evaluation_result = evaluate_fact_with_ai(fact_to_check, top_snippets)
                    
                    st.subheader("⚖️ حكم منصة التحقق:")
                    if "صحيح" in evaluation_result and "جزئي" not in evaluation_result:
                        st.success(evaluation_result)
                    elif "جزئي" in evaluation_result:
                        st.warning(evaluation_result)
                    else:
                        st.error(evaluation_result)
