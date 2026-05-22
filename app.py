import streamlit as st
import requests
from datetime import datetime
import os
import json
import urllib.parse
from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from supabase import create_client, Client

# --- جلب المفاتيح البرمجية من بيئة التشغيل (Secrets) ---
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
SERPER_API_KEY = os.environ.get('SERPER_API_KEY')
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

def apply_arabic_rtl():
    """حقن كود CSS لقلب واجهة التطبيق بالكامل لتصبح من اليمين إلى اليسار وتغيير الخط ليكون مريحاً للعين"""
    st.markdown(
        """
        <style>
        /* إجبار التطبيق بالكامل على الاتجاه من اليمين إلى اليسار */
        .stApp {
            direction: rtl;
            text-align: right;
        }
        
        /* ضبط صناديق النصوص ومدخلات المستخدم لتكون محاذاتها يميناً */
        div[data-baseweb="textarea"] textarea {
            direction: rtl !important;
            text-align: right !important;
        }
        div[data-baseweb="input"] input {
            direction: rtl !important;
            text-align: right !important;
        }
        
        /* ضبط نصوص قوقل والمحتويات المقتبسة لتلتزم باليمين */
        .stMarkdown div p {
            direction: rtl;
            text-align: right;
        }
        
        /* تحسين شكل وحواف صناديق التنبيه (الأخضر والأحمر والأصفر) */
        .stAlert {
            direction: rtl;
            text-align: right;
        }
        
        /* محاذاة العناوين الرئيسية والفرعية */
        h1, h2, h3, h4, h5, h6, p, span {
            text-align: right !important;
            direction: rtl !important;
        }
        
        /* تعديل اتجاه صناديق الأرشيف القابلة للتوسيع (Expander) */
        .st-emotion-cache-p6w706 {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# قائمة المصادر الموثوقة (Tier 2)
TRUSTED_DOMAINS = [
    "bbc.com", "reuters.com", "cnn.com", "skynewsarabia.com", 
    "aljazeera.net", "france24.com", "asharq.com", "alarabiya.net",
    "dw.com", "un.org", "who.int", "reutersagency.com"
]

# --- تهيئة المكونات وقاعدة البيانات في الكاش لسرعة الأداء ---
@st.cache_resource
def init_services():
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # تهيئة عميل Supabase
    supabase_client = None
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            st.error(f"خطأ أثناء الاتصال بقاعدة بيانات Supabase: {e}")
            
    return groq_client, embed_model, supabase_client

groq_client, embed_model, supabase_client = init_services()

# --- دالات الحفظ والقراءة من قاعدة البيانات المحدثة لكشف الأخطاء ---
def save_check_to_database(fact, verdict, final_answer):
    """حفظ التدقيق الجديد في Supabase مع طباعة الخطأ إن وجد"""
    if supabase_client:
        try:
            data, count = supabase_client.table("fact_checks").insert({
                "fact": fact,
                "verdict": verdict,
                "final_answer": final_answer
            }).execute()
            # لإنعاش الصفحة فوراً بعد الحفظ لتظهر النتيجة في الأرشيف
            #st.rerun() 
        except Exception as e:
            # هذا السطر سيطبع لك السبب الحقيقي للخطأ في الـ Manage App / Logs الخاصة بـ Streamlit
            st.sidebar.error(f"فشل حفظ البيانات: {e}")

def get_recent_checks(limit=5):
    """جلب آخر التدقيقات من Supabase"""
    if supabase_client:
        try:
            response = supabase_client.table("fact_checks").select("*").order("created_at", desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            st.sidebar.error(f"فشل قراءة الأرشيف: {e}")
            return []
    return []

# --- الدالات المساعدة للبحث والكشط والذكاء الاصطناعي ---
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-8)

def filter_and_rank_sources(fact, snippets, top_k=3):
    if not embed_model or not snippets:
        return snippets[:top_k]
    vectors = embed_model.encode([s['text'] for s in snippets])
    fact_vector = embed_model.encode([fact])[0]
    for i, snippet in enumerate(snippets):
        base_conf = float(cosine_similarity(fact_vector, vectors[i]))
        source_url = snippet['source'].lower()
        if any(social in source_url for social in ["facebook.com", "fb.com", "twitter.com", "x.com", "tiktok.com", "instagram.com"]):
            base_conf -= 0.25
        elif any(trusted in source_url for trusted in TRUSTED_DOMAINS):
            base_conf += 0.15
        snippets[i]['confidence'] = base_conf
    return sorted(snippets, key=lambda x: x['confidence'], reverse=True)[:top_k]

def search_trusted_sources_serper(query, api_key, num_results=3):
    if not api_key:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    data = {"q": query, "num": num_results}
    snippets = []
    try:
        response = requests.post(url, headers=headers, json=data)
        results = response.json()
        for item in results.get("organic", []):
            snippet_text = item.get("snippet")
            link = item.get("link")
            if snippet_text:
                snippets.append({"text": snippet_text, "source": link})
        return snippets
    except:
        return []

def scrape_full_content(target_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(target_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
        full_text = " ".join([txt.get_text().strip() for txt in text_elements if txt.get_text().strip()])
        return full_text[:1200]  # حمية التوكنز الآمنة لمنع خطأ 413
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
        for preferred in ["qwen", "llama-3.3"]:
            match = next((m for m in available_models if preferred in m.lower() and "preview" not in m.lower()), None)
            if match: return match
        return available_models[0] if available_models else None
    except:
        return None

def extract_source_entity(fact):
    model = get_active_model()
    if not model or not groq_client: return None, None
    prompt = f'تحلل النص واستخرج منه أي جهة نُسب إليها الكلام. أعد الإجابة بصيغة JSON فقط: {{"has_entity": true, "entity_name": "الاسم", "expected_domain": "who.int"}}. النص: "{fact}"'
    try:
        response = groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        data = json.loads(response.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        if data.get("has_entity"): return data.get("entity_name"), data.get("expected_domain")
    except: pass
    return None, None

def parse_ai_response(full_text):
    if "<think>" in full_text and "</think>" in full_text:
        parts = full_text.split("</think>")
        return parts[0].replace("<think>", "").strip(), parts[1].strip()
    return None, full_text

def evaluate_fact_with_multi_tier(fact, tier1, tier2, tier3, entity_name):
    model = get_active_model()
    if not model: return "خطأ في الاتصال بالنموذج"
    
    def compile_context(sources, label):
        return "\n\n".join([f"[{label}: {s['source']}]\nالمحتوى: {scrape_full_content(s['source']) or s['text']}" for s in sources[:2]])

    c1 = compile_context(tier1, f"موقع {entity_name}")
    c2 = compile_context(tier2, "وكالة أنباء موثوقة")
    c3 = compile_context(tier3, "الويب العام")
    
    prompt = f"""أنت رئيس تحرير ومحقق صحفي خبير. تاريخ اليوم الحالي: {get_current_live_date()}. 
الادعاء: "{fact}". موازنة الأدلة بناءً على المستندات المرفقة:
المستوى 1: {c1}
المستوى 2: {c2}
المستوى 3: {c3}

🛑 [تعليمات صارمة للرد]:
يجب أن تبدأ ردك بوضع تصنيف قاطع وحيد للادعاء بين هذه الأقواس الثلاثة فقط:
Either [VERDICT: TRUE] or [VERDICT: FALSE] or [VERDICT: PARTIAL]
ثم بعد هذا الوسم، اكتب تفكيكك والتحليل الكامل والبديل الحقيقي باللغة العربية براحتك."""
    
    try:
        response = groq_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"خطأ: {e}"

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
st.set_page_config(page_title="المُحقق الذكي", layout="centered")
# 🔥 تفعيل التصميم العربي المندمج فوراً عند فتح التطبيق
apply_arabic_rtl()
st.header("🛡️ المُحقق الذكي")
st.caption(f"📅 تاريخ التحقق: {get_current_live_date()}")

fact_to_check = st.text_area("أدخل المعلومة أو الخبر المراد فحصه:", "")

if st.button("بدء الفحص الجنائي الرقمي"):
    if fact_to_check.strip() == "":
        st.warning("الرجاء كتابة نص أولاً.")
    else:
        tier1_sources, tier2_sources, tier3_sources = [], [], []
        
        with st.spinner("🕵️ جاري تفكيك الادعاء وفق 3 طبقات ..."):
            entity_name, expected_domain = extract_source_entity(fact_to_check)
            if entity_name and expected_domain:
                tier1_sources = search_trusted_sources_serper(f"site:{expected_domain} {fact_to_check}", SERPER_API_KEY, num_results=2)
            
            sites_query = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS[:4]])
            tier2_sources = search_trusted_sources_serper(f"({sites_query}) {fact_to_check}", SERPER_API_KEY, num_results=2)
            
            raw_tier3 = search_trusted_sources_serper(f"{fact_to_check} {datetime.now().year}", SERPER_API_KEY, num_results=3)
            tier3_sources = filter_and_rank_sources(fact_to_check, raw_tier3, top_k=2)

        if not tier1_sources and not tier2_sources and not tier3_sources:
            st.warning("لم نتمكن من جلب أدلة حية كافية.")
        else:
            evaluation_result = evaluate_fact_with_multi_tier(fact_to_check, tier1_sources, tier2_sources, tier3_sources, entity_name)
            thinking, final_answer = parse_ai_response(evaluation_result)
            
            if thinking:
                with st.expander("🧠 مذكرات التحليل الداخلي للمحقق (Chain of Thought):"):
                    st.write(thinking)
            
            st.subheader("⚖️ حكم منصة التحقق النهائي:")
            
            # استخراج الحكم الصارم من الوسم وتحديد اللون بدقة 100%
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
            else:
                # حل احتياطي لو لم يلتزم الموديل بالوسم (فحص تقليدي)
                if "جزئي" in final_answer:
                    verdict_type = "جزئيًا صحيح"
                    st.warning(final_answer)
                elif "صحيح" in final_answer:
                    verdict_type = "صحيح"
                    st.success(final_answer)
                else:
                    st.error(final_answer)
            
            # حفظ التحقيق الحالي بدقة التصنيف الجديدة في قاعدة البيانات
            save_check_to_database(fact_to_check, verdict_type, clean_answer)
            
            # عرض أزرار المشاركة والنسخ بالنص النظيف
            display_share_buttons(fact_to_check, clean_answer)
            st.markdown(" ")
            st.code(f"الادعاء: {fact_to_check}\nالحكم النهائي: {clean_answer}", language="text")

# --- 🔥 قسم السجل العام (آخر التدقيقات الحديثة) ---
st.markdown("---")
st.subheader("🔔 آخر الشائعات التي تم تفكيكها حديثاً عبر المنصة:")
recent_items = get_recent_checks()

if recent_items:
    for item in recent_items:
        # تحديد اللون الأيقوني حسب الحكم المخزن في Supabase
        badge = "🔴" if "خاطئ" in item['verdict'] else ("🟡" if "جزئي" in item['verdict'] else "🟢")
        with st.expander(f"{badge} {item['fact'][:70]}..."):
            st.markdown(f"**الادعاء الأصلي:** {item['fact']}")
            st.markdown(f"**الحكم والتحليل:** {item['final_answer']}")
            st.caption(f"📅 تم التدقيق في: {item['created_at'][:10]}")
else:
    st.info("لا توجد تدقيقات سابقة مسجلة في الأرشيف حتى الآن.")
