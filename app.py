import streamlit as st
import requests
from datetime import datetime
import os
import re
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

def search_trusted_sources_serper(fact, api_key, num_results=5):
    if not api_key:
        st.error("خطأ: مفتاح SERPER_API_KEY غير متوفر.")
        return []

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    
    # تحسين الاستعلام بربطه بالزمن الحالي تلقائياً لضمان جلب أخبار حية محدثة
    current_year = datetime.now().year
    enhanced_query = f"{fact} {current_year}"
    
    data = {"q": enhanced_query, "num": num_results}

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
    except Exception as e:
        st.error(f"خطأ أثناء جلب البيانات من Serper: {e}")
        return []

def scrape_full_content(target_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(target_url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
        full_text = " ".join([txt.get_text().strip() for txt in text_elements if txt.get_text().strip()])
        return full_text[:3500]
    except:
        return ""

def get_current_live_date():
    months_ar = {
        1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
        7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"
    }
    now = datetime.now()
    return f"{now.day} {months_ar[now.month]} {now.year}"

def parse_ai_response(full_text):
    """
    دالة تقوم بفصل وعزل وسم التفكير الداخلي للنموذج <think> عن الحكم النهائي الموجه للمستخدم
    """
    if "<think>" in full_text and "</think>" in full_text:
        parts = full_text.split("</think>")
        thinking_process = parts[0].replace("<think>", "").strip()
        final_answer = parts[1].strip()
        return thinking_process, final_answer
    elif "</think>" in full_text:
        parts = full_text.split("</think>")
        return parts[0].strip(), parts[1].strip()
    return None, full_text

def evaluate_fact_with_ai(fact, top_snippets):
    if not groq_client:
        return "خطأ: نموذج Groq غير مهيأ، تحقق من إضافة مفتاح GROQ_API_KEY بنجاح."
    if not top_snippets:
        return "لا توجد معلومات كافية لتقييم المعلومة."

    scraped_contexts = []
    for index, s in enumerate(top_snippets):
        with st.spinner(f"جاري قراءة وتحليل المصدر رقم {index+1} بالكامل..."):
            full_body = scrape_full_content(s['source'])
            content_to_use = full_body if full_body else s['text']
            scraped_contexts.append(f"[المصدر: {s['source']}]\nالمحتوى: {content_to_use}")

    context = "\n\n---\n\n".join(scraped_contexts)
    
    # جلب التاريخ اللحظي الفعلي
    live_date = get_current_live_date()
    
    prompt = f"""
أنت محقق صحفي خبير وصارم جداً مخصص للتحقق من الحقائق والأخبار (Fact-Checker Agent).

🛑 [سياق زمني حرج للغاية]: تاريخ اليوم الحالي هو بالضبط: {live_date}. 
يجب أن تحاكم وتفحص كل التواريخ المذكورة في المصادر بناءً على هذا التاريخ الصارم (مثال: إذا كنا في عام 2026 والمقال يتحدث عن حدث في 2024، فهذا حدث من الماضي).

⚙️ [منهجية التفكير المطلوبة منك لتفادي الأخطاء]:
عند فحص الادعاء، يجب أن تتبع الخطوات التالية في عقلك وتفكيرك:
1. تفكيك الادعاء لعناصره (من، ماذا، ومتى).
2. إذا تبين لك أن الادعاء "خاطئ"، لا تكتفي بكلمة خاطئ بل قم بـ "التحقيق العكسي": ابحث في المصادر عمن هو صاحب الصفة الحقيقية الحالية في هذا الزمن المذكور.
3. صياغة الحكم النهائي للمستخدم بوضوح شديد.

💡 أمثلة لطريقة الصياغة والتفكير المطلوبة:
- الادعاء: "الأهلي بطلاً للدوري المصري 2026"
- طريقة ردك: "خاطئ، بناءً على البيانات الرسمية الحالية لعام 2026 فإن بطل الدوري هو [اسم النادي الحقيقي من المصادر]، بينما الأهلي حل في المركز الثاني."

---

الادعاء المراد فحصها الآن:
"{fact}"

المصادر المستخرجة الحية من الويب:
{context}

قم بالتحقق الآن وصغ الحكم والسبب بدقة بناءً على القواعد السابقة باللغة العربية:
"""

    try:
        # جلب النماذج الحية تلقائياً واختيار الأنسب
        available_models = [m.id for m in groq_client.models.list().data]
        selected_model = None
        for preferred in ["qwen", "llama-3.3", "llama3-70b"]:
            match = next((m for m in available_models if preferred in m.lower() and "preview" not in m.lower()), None)
            if match:
                selected_model = match
                break
        
        if not selected_model and available_models:
            selected_model = available_models[0]

        response = groq_client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # درجة منخفضة جداً لضمان الالتزام الصارم بالحقائق والمصادر
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"حدث خطأ أثناء استدعاء نموذج جروك: {e}"

# --- واجهة مستخدم Streamlit ---
st.set_page_config(page_title="مُحقق الحقائق الذكي", layout="centered")
st.header("🔍 نظام التأكد من الحقائق الذكي (إصدار العميل المحقق)")

# عرض التاريخ الحالي في الواجهة لإعلام المستخدم
st.caption(f"📅 تاريخ النظام الحي والمستخدم في الفحص اليوم: {get_current_live_date()}")

fact_to_check = st.text_area("أدخل المعلومة أو الخبر المراد فحصه بدقة:", "الرئيس الأمريكي الحالي في 2026 هو دونالد ترامب")

if st.button("بدء فحص وتفكيك الحقيقة"):
    if not GROQ_API_KEY or not SERPER_API_KEY:
        st.error("المفاتيح البرمجية ناقصة في إعدادات Secrets.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أولاً.")
    else:
        with st.spinner("جاري كشط محرك البحث وقراءة الروابط الحية..."):
            snippets = search_trusted_sources_serper(fact_to_check, SERPER_API_KEY)
            
            if not snippets:
                st.warning("لم نتمكن من العثور على مصادر ويب حديثة وموثوقة متعلقة بهذا النص.")
            else:
                top_snippets = filter_top_snippets(fact_to_check, snippets, top_k=3)
                
                st.subheader("📌 المصادر الحية المعتمد عليها في القراءة الكاملة:")
                for s in top_snippets:
                    date_str = s['date'].strftime("%Y-%m-%d") if s['date'] else "تاريخ غير معلوم"
                    st.markdown(f"- [{s['source']}]({s['source']}) *({date_str})*")
                
                with st.spinner("يقوم العميل الذكي الآن بتفكيك الادعاء ومحاكمة التواريخ..."):
                    evaluation_result = evaluate_fact_with_ai(fact_to_check, top_snippets)
                    
                    # فصل التفكير عن الإجابة النهائية
                    thinking, final_answer = parse_ai_response(evaluation_result)
                    
                    # إذا كان النموذج يحتوي على تفكير داخلي (مثل كوين) نعرضه في كاشف أنيق ومنسدل
                    if thinking:
                        with st.expander("🧠 رؤية مسار خطة وتفكيك النموذج الداخلي (Chain of Thought):"):
                            st.write(thinking)
                    
                    st.subheader("⚖️ حكم منصة التحقق:")
                    if "صحيح" in final_answer and "جزئي" not in final_answer:
                        st.success(final_answer)
                    elif "جزئي" in final_answer:
                        st.warning(final_answer)
                    else:
                        st.error(final_answer)
