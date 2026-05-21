import streamlit as st
import requests
from datetime import datetime
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# جلب المفاتيح من بيئة التشغيل
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
SERPER_API_KEY = os.environ.get('SERPER_API_KEY')

# تهيئة وتحميل النماذج لمرة واحدة وحفظها في الكاش لسرعة الأداء
@st.cache_resource
def load_models():
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # التعديل هنا: إضافة بادئة /models لحل مشكلة الـ 404
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
    else:
        gemini_model = None
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return gemini_model, embed_model

gemini_model, embed_model = load_models()

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

def search_trusted_sources_serper(fact, api_key, num_results=10, recent_year=2020):
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

def evaluate_fact_gemini(fact, top_snippets):
    if not gemini_model:
        return "خطأ: نموذج Gemini غير مهيأ، تحقق من الرمز الخاص بك."
    if not top_snippets:
        return "لا توجد معلومات كافية لتقييم المعلومة."

    context = "\n".join([f"- {s['text']}" for s in top_snippets])
    prompt = f"""
    أنت مساعد خبير ومحايد للتحقق من الحقائق باللغة العربية.
    بناءً على المصادر الموثوقة المرفقة فقط، قم بتقييم المعلومة التالية.
    
    المعلومة: "{fact}"
    المصادر:
    {context}

    أجب بدقة وبصياغة مختصرة جداً تبدأ بـ: "صحيح"، "جزئيًا صحيح"، أو "خاطئ"، ثم اذكر السبب المباشر في جملة واحدة إضافية.
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# --- واجهة مستخدم Streamlit ---
st.set_page_config(page_title="مُحقق الحقائق الذكي", layout="centered")
st.header("🔍 نظام التحقق من الحقائق الذكي ")

fact_to_check = st.text_area("أدخل المعلومة المراد فحصها:", "السرطان يقوي عظام المريض")

if st.button("بدء فحص الحقيقة"):
    if not GEMINI_API_KEY or not SERPER_API_KEY:
        st.error("المفاتيح البرمجية ناقصة. تأكد من إضافتها في قسم Secrets بكولاب وإعادة تشغيل التطبيق.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أو معلومة أولاً للفحص.")
    else:
        with st.spinner("جاري البحث عبر الويب وتصفية النتائج الذكية..."):
            snippets = search_trusted_sources_serper(fact_to_check, SERPER_API_KEY)
            
            if not snippets:
                st.warning("لم نتمكن من العثور على مصادر ويب حديثة وموثوقة متعلقة بهذا النص.")
            else:
                top_snippets = filter_top_snippets(fact_to_check, snippets, top_k=3)
                
                st.subheader("📌 أهم المصادر والمقتبسات المستند إليها:")
                for s in top_snippets:
                    date_str = s['date'].strftime("%Y-%m-%d") if s['date'] else "تاريخ غير معلوم"
                    st.markdown(f"- [{s['source']}]({s['source']}): {s['text']} *({date_str})*")
                
                with st.spinner("جاري صياغة التقييم النهائي بواسطة Gemini..."):
                    evaluation_result = evaluate_fact_gemini(fact_to_check, top_snippets)
                    
                    st.subheader("⚖️ حكم منصة التحقق:")
                    if "صحيح" in evaluation_result and "جزئي" not in evaluation_result:
                        st.success(evaluation_result)
                    elif "جزئي" in evaluation_result:
                        st.warning(evaluation_result)
                    else:
                        st.error(evaluation_result)
