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

# --- قائمة المصادر الموثوقة ذات الأولوية العالية (Tier 2) ---
TRUSTED_DOMAINS = [
    "bbc.com", "reuters.com", "cnn.com", "skynewsarabia.com", 
    "aljazeera.net", "france24.com", "asharq.com", "alarabiya.net",
    "dw.com", "un.org", "who.int", "reutersagency.com"
]

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

def filter_and_rank_sources(fact, snippets, top_k=4):
    """
    دالة متطورة لترتيب المصادر: تجمع بين الشبه الدلالي (Semantic Embedding)
    وبين معاقبة وخفض ترتيب شبكات التواصل الاجتماعي لرفع مصداقية المواقع الرسمية.
    """
    if not embed_model or not snippets:
        return snippets[:top_k]
    
    vectors = embed_model.encode([s['text'] for s in snippets])
    fact_vector = embed_model.encode([fact])[0]

    for i, snippet in enumerate(snippets):
        base_conf = float(cosine_similarity(fact_vector, vectors[i]))
        source_url = snippet['source'].lower()
        
        # عقوبة جزائية لمنصات التواصل الاجتماعي لخفض ترتيبها تلقائياً
        if any(social in source_url for social in ["facebook.com", "fb.com", "twitter.com", "x.com", "tiktok.com", "instagram.com"]):
            base_conf -= 0.25  # خفض الأولوية بشكل حاد
        # مكافأة تشجيعية للمصادر الموثوقة المحددة مسبقاً
        elif any(trusted in source_url for trusted in TRUSTED_DOMAINS):
            base_conf += 0.15  # رفع الأولوية
            
        snippets[i]['confidence'] = base_conf

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
    model = get_active_model()
    if not model or not groq_client:
        return None, None
        
    prompt = f"""
    حلل النص واستخرج منه أي إشارة لصحيفة، موقع، منظمة، أو جهة نُسب إليها الكلام (مثل: منظمة الصحة العالمية، اليوم السابع، bbc، رويترز).
    أعد الإجابة بصيغة JSON فقط كالتالي دون أي نص إضافي:
    {{
      "has_entity": true أو false,
      "entity_name": "اسم الجهة المستخرجة بالعربية",
      "expected_domain": "الدومين التقريبي للموقع مثل who.int أو bbc.com أو reuters.com"
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

def build_trusted_query(fact):
    """بناء استعلام مخصص للبحث حصراً داخل المواقع ذات الثقة العالية (Tier 2)"""
    sites_query = " OR ".join([f"site:{domain}" for domain in TRUSTED_DOMAINS[:6]]) # دمج أول 6 نطاقات كبار لعدم تجاوز طول الاستعلام
    return f"({sites_query}) {fact}"

def evaluate_fact_with_multi_tier(fact, tier1_sources, tier2_sources, tier3_sources, entity_name):
    model = get_active_model()
    if not model:
        return "خطأ: لم يتم العثور على نماذج نشطة في حساب جروك."

    # دالة فرعية لتجميع وتجهيز نصوص الكشط بشكل منظم للنموذج
    def compile_context(sources, label):
        compiled = []
        for s in sources:
            body = scrape_full_content(s['source'])
            compiled.append(f"[{label}: {s['source']}]\nالمحتوى: {body if body else s['text']}")
        return "\n\n---\n\n".join(compiled) if compiled else "لا تتوفر مستندات كافية في هذا المستوى."

    context_tier1 = compile_context(tier1_sources, f"مستند من المصدر المنسوب إليه مباشرة ({entity_name})")
    context_tier2 = compile_context(tier2_sources, "وثيقة من منصة إعلامية ذات مصداقية عالية (BBC, Reuters, etc.)")
    context_tier3 = compile_context(tier3_sources, "منشور أو تغطية من الويب العام")

    live_date = get_current_live_date()
    
    prompt = f"""
أنت رئيس تحرير ومحقق صحفي خبير في كشف الزيف (Lead Fact-Checking Auditor).
🛑 [سياق زمني]: تاريخ اليوم الحالي هو: {live_date}. 

الادعاء المطلوب فصحه ومحاكمته: "{fact}"

لقد قمنا بجمع الأدلة لك بناءً على 3 مستويات صارمة من الثقة، وعليك الموازنة بينها (تذكر: وثيقة رسمية من رويترز أو موقع المنظمة تلغي وتكذب أي منشور على فيسبوك أو مدونة شخصية مجهولة):

📂 [الأدلة من المستوى الأول - المصدر المذكور في الادعاء مباشرة]:
{context_tier1}

📂 [الأدلة من المستوى الثاني - وكالات الأنباء العالمية الموثوقة المحددة مسبقاً]:
{context_tier2}

📂 [الأدلة من المستوى الثالث - الويب العام وشبكات التواصل]:
{context_tier3}

⚙️ قواعد المحاكمة وإصدار الحكم:
1. حلل التناقضات: إذا وجد إعلان منتشر بالويب العام (المستوى 3) ولكن أرشيف الوكالات الموثوقة (المستوى 2) والمصدر نفسه (المستوى 1) يخلو تماماً منه أو ينفيه، فاحكم فوراً بأنه "خاطئ" ووضح للمستخدم الثغرة.
2. ابدأ بكلمة: "صحيح"، "جزئيًا صحيح"، أو "خاطئ"، متبوعاً بالتفكيك والبديل الفعلي لعام {datetime.now().year}.
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
st.set_page_config(page_title="مُحقق الحقائق الذكي السيادي", layout="centered")
st.header("🛡️ نظام التأكد من الحقائق الذكي (نظام مستويات الثقة الثلاثة)")
st.caption(f"📅 تاريخ النظام اللحظي: {get_current_live_date()}")

fact_to_check = st.text_area("أدخل المعلومة أو الخبر المراد فحصه ومحاكمته برمتها:", "قالت وكالة رويترز إن السيرفرات العالمية ستتوقف بالكامل غداً")

if st.button("بدء الفحص الجنائي الرقمي"):
    if not GROQ_API_KEY or not SERPER_API_KEY:
        st.error("المفاتيح البرمجية ناقصة في إعدادات Secrets.")
    elif fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أولاً.")
    else:
        current_year = datetime.now().year
        
        tier1_sources = []
        tier2_sources = []
        tier3_sources = []
        
        # 1. تشغيل المستوى الأول (التقصي عن الجهة المذكورة)
        with st.spinner("🕵️ الطبقة 1: فحص الادعاء واستخراج الجهة المنسوب إليها..."):
            entity_name, expected_domain = extract_source_entity(fact_to_check)
            if entity_name and expected_domain:
                st.toast(f"تم رصد الكيان: {entity_name}، جاري فحص أرشيفه المباشر...", icon="🏢")
                tier1_sources = search_trusted_sources_serper(f"site:{expected_domain} {fact_to_check}", SERPER_API_KEY, num_results=2)

        # 2. تشغيل المستوى الثاني (البحث في Whitelist الوكالات الكبرى الموثوقة)
        with st.spinner("🛡️ الطبقة 2: التفتيش المتوازي في وكالات الأنباء العالمية الموثوقة..."):
            trusted_query = build_trusted_query(fact_to_check)
            tier2_sources = search_trusted_sources_serper(trusted_query, SERPER_API_KEY, num_results=3)

        # 3. تشغيل المستوى الثالث (البحث المفتوح بالويب العام)
        with st.spinner("🌐 الطبقة 3: تجميع البيانات المفتوحة ومعاقبة مصادر التواصل الاجتماعي..."):
            raw_tier3 = search_trusted_sources_serper(f"{fact_to_check} {current_year}", SERPER_API_KEY, num_results=5)
            # استخدام الفلتر والترتيب الذكي لخفض رتبة الفيسبوك وإكس وتفضيل المقالات الرسمية
            tier3_sources = filter_and_rank_sources(fact_to_check, raw_tier3, top_k=3)

        # التأكد من وجود أي داتا قبل المضي قدماً
        if not tier1_sources and not tier2_sources and not tier3_sources:
            st.warning("لم نتمكن من جلب أدلة ويب حية كافية لمطابقة هذا النص.")
        else:
            # عرض هيكل وشجرة المصادر المكتشفة بشفافية للمستخدم في الواجهة
            st.subheader("📊 شجرة المصادر والأدلة التي تم جمعها ومحاكمتها:")
            
            if entity_name:
                st.markdown(f"**🏢 المستوى 1 ({entity_name}):** {f'تم العثور على {len(tier1_sources)} مستندات داخل موقعهم' if tier1_sources else 'لم ينشروا هذا النص إطلاقاً بصفحتهم الرسمية'}")
                
            st.markdown(f"**🛡️ المستوى 2 (الوكالات الموثوقة الكبرى):** تم تأمين {len(tier2_sources)} روابط موثقة من (BBC, Reuters...)")
            st.markdown(f"**🌐 المستوى 3 (الويب العام):** تم رصد وتصفية {len(tier3_sources)} مقالات مع موازنة الأوزان.")
            
            # 4. صياغة وتقييم النتيجة النهائية عبر المحقق الآلي
            with st.spinner("⚖️ يقوم رئيس التحرير الآلي الآن بموازنة قوة الأدلة وإصدار الحكم..."):
                evaluation_result = evaluate_fact_with_multi_tier(fact_to_check, tier1_sources, tier2_sources, tier3_sources, entity_name)
                
                # فصل التفكير وعرض النتيجة بشكل رصين
                thinking, final_answer = parse_ai_response(evaluation_result)
                
                if thinking:
                    with st.expander("🧠 مذكرات التفكير والتحليل الداخلي لرئيس التحرير (Chain of Thought):"):
                        st.write(thinking)
                
                st.subheader("⚖️ حكم منصة التحقق النهائي:")
                if "صحيح" in final_answer and "جزئي" not in final_answer:
                    st.success(final_answer)
                elif "جزئي" in final_answer:
                    st.warning(final_answer)
                else:
                    st.error(final_answer)
