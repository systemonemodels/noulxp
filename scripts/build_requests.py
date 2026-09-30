"""Write src/noulxp/data/requests-0.1.jsonl: the fixed request set conformance files are made from.

The set is deterministic and hand-written: every question type, 2 to 20
options, bare and described options, instructions in several languages,
short and long states, and the edge cases the profiles define (reserved
marker text, an option over the per-option budget, >10 options, a state
ending in a newline, an empty state, an empty noul instruction).

    python scripts/build_requests.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "noulxp" / "data" / "requests-0.1.jsonl"


def choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul(instructions, criteria=None):
    q = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return q


def support_thread(turns: int) -> str:
    """A long, varied support conversation (English)."""
    lines = []
    for i in range(turns):
        day = 3 + i
        lines.append(
            f"Customer (March {day}): I am writing again about order #{1182 + i % 3}. "
            f"The card ending in 44{i % 10}1 was charged twice, once for {49 + i}.90 and once "
            f"for the same amount a few minutes later. My bank says the second charge is "
            f"final unless the merchant reverses it."
        )
        lines.append(
            f"Agent (March {day}): Thank you for your patience. I have forwarded case "
            f"{7300 + 17 * i} to the payments team; they usually answer within two working "
            f"days. Could you confirm the last four digits of the card and the email on the "
            f"account?"
        )
        if i % 2 == 0:
            lines.append(
                "Customer: I already sent those details twice. Nobody has called me back, the "
                "chat closes every time I ask for a supervisor, and the refund form shows an "
                "error when I upload the bank statement."
            )
    return "\n".join(lines)


NEPALI_LONG = " ".join(
    [
        "हाम्रो गाउँमा पक्की सडक बनेको तीन वर्ष भयो, तर यो वर्षको मनसुनले सडकका धेरै भाग भत्काइदियो।",
        "विद्यालय जाने बालबालिकाहरू अहिले खोला तरेर जानुपर्छ, र बिहान पानी धेरै हुँदा उनीहरू घरमै बस्छन्।",
        "गाउँपालिकाले मर्मतका लागि बजेट छुट्याएको भनेको छ, तर काम अझै सुरु भएको छैन।",
        "स्थानीय स्वास्थ्य चौकीमा औषधि पुर्‍याउन पनि ढिलो भइरहेको छ, किनकि गाडी बीच बाटोमै अड्किन्छन्।",
        "किसानहरूले तरकारी बजारसम्म पुर्‍याउन नसक्दा धेरै उत्पादन खेर गइरहेको छ।",
        "युवाहरूले श्रमदान गरेर अस्थायी पुल बनाए, तर ठूलो पानी आउँदा त्यो पनि बग्ने डर छ।",
    ]
)

THAI_LONG = " ".join(
    [
        "เช้าวันนี้การจราจรบนถนนพระรามสี่ติดขัดอย่างหนักตั้งแต่เจ็ดโมงเช้า",
        "เนื่องจากมีรถบรรทุกเสียกลางสะพานและฝนตกหนักต่อเนื่องตั้งแต่เมื่อคืน",
        "ผู้โดยสารรถประจำทางหลายสายต้องลงเดินเท้าเพื่อไปต่อรถไฟฟ้า",
        "เจ้าหน้าที่ตำรวจจราจรแนะนำให้หลีกเลี่ยงเส้นทางดังกล่าวจนถึงช่วงบ่าย",
        "ขณะที่กรมอุตุนิยมวิทยาคาดว่าจะมีฝนตกหนักอีกครั้งในช่วงเย็น",
        "โรงเรียนบางแห่งในเขตคลองเตยประกาศให้นักเรียนเข้าเรียนช้ากว่าปกติหนึ่งชั่วโมง",
    ]
)

MEETING_NOTES = (
    "Weekly sync, product team. Attendees: Ana, Bilal, Chen, Dora. "
    "1) The March release slipped by one week because the payment provider changed its API "
    "without notice; Bilal will write the migration and Chen will review it by Thursday. "
    "2) Support tickets about duplicate charges doubled since Monday; Dora thinks it is the "
    "retry logic in checkout and will add logging today. "
    "3) Marketing wants the new pricing page live before the webinar on the 21st; Ana will "
    "confirm copy with legal. "
    "4) Nobody has owned the flaky end-to-end tests for two sprints; we agreed to rotate the "
    "duty weekly, starting with Chen. Next meeting: Tuesday 10:00."
)

LONG_OPTION = (
    "the customer describes a recurring technical failure that affects several devices in "
    "the household, has already tried restarting the router, resetting the password and "
    "reinstalling the application, and asks for an engineer visit together with compensation "
    "for the days without service and a written explanation of the root cause"
)

REQUESTS = [
    (
        "r01",
        ["lang:en", "multi"],
        "Customer: I was charged twice for my order last week and I still haven't got a refund. This is the third time I'm writing.",
        {
            "intent": choice(
                "What does the customer want?",
                {"refund": None, "cancel order": None, "track delivery": None, "other": None},
            ),
            "urgency": score("How urgent is this?", ["low", "medium", "high"]),
            "angry": noul("The customer is angry."),
        },
    ),
    (
        "r02",
        ["lang:en"],
        "Hi, I can't log in since I changed my phone. The reset link never arrives in my inbox.",
        {
            "team": choice(
                "Which team should handle this request?",
                {
                    "billing": "Billing and payment disputes",
                    "access": "Account access and login",
                    "shipping": "Shipping and delivery",
                },
            ),
            "needs_human": noul(
                "Does a person need to handle this?",
                {"true": "a person must step in", "false": "an automated reply is enough"},
            ),
        },
    ),
    (
        "r03",
        ["lang:en", "short-state"],
        "hi",
        {
            "intent": choice("What is the user doing?", ["greeting", "question", "complaint"]),
            "help": noul("Is the user asking for help?"),
        },
    ),
    (
        "r04",
        ["lang:en"],
        "Battery lasts two full days, the screen is gorgeous and setup took five minutes. Only complaint: the charger is sold separately.",
        {
            "sentiment": score(
                "How positive is this review?",
                ["very negative", "negative", "neutral", "positive", "very positive"],
            ),
            "recommend": noul("Would the reviewer recommend the product?"),
        },
    ),
    (
        "r05",
        ["lang:en", "numbered-levels"],
        "After the last deploy, checkout returns a 500 error for every customer paying by card. Logs show a null pointer in the payment service.",
        {
            "severity": score(
                "How severe is this bug?", ["0: cosmetic", "1: minor", "2: major", "3: critical"]
            ),
            "component": choice(
                "Which component is failing?",
                {
                    "frontend": "web UI and pages",
                    "backend": "API and services",
                    "database": "storage and queries",
                    "auth": "login and permissions",
                    "payments": "checkout and billing",
                    "infra": "deploys, servers and networking",
                },
            ),
        },
    ),
    (
        "r06",
        ["lang:en"],
        "Hello, I'm a journalist at the Daily Ledger writing about remote work. Could someone from your team comment on your new four-day week policy by Friday?",
        {
            "department": choice(
                "Which department should receive this email?",
                {
                    "sales": "buying the product",
                    "support": "help with the product",
                    "billing": "invoices and payments",
                    "legal": "contracts and compliance",
                    "hr": "jobs and employees",
                    "press": "journalists and media",
                    "partnerships": "integrations and resellers",
                    "spam": "unsolicited or irrelevant",
                },
            ),
        },
    ),
    (
        "r07",
        ["lang:en"],
        "You are all idiots and whoever wrote this update should be fired. Worst app ever.",
        {
            "toxic": noul("Is this message abusive?", {"true": "insults, threats or harassment"}),
            "category": choice(
                "How should moderation label it?",
                ["spam", "harassment", "hate", "self-promotion", "fine"],
            ),
        },
    ),
    (
        "r08",
        ["lang:en"],
        MEETING_NOTES,
        {
            "action_needed": noul("Does anyone need to do something after this meeting?"),
            "priority": score("How pressing are the open items?", ["low", "normal", "high"]),
        },
    ),
    (
        "r09",
        ["lang:en"],
        "The change adds input validation but the new function has no tests, and it swallows every exception with a bare except.",
        {
            "verdict": choice(
                "What should the reviewer do?",
                {
                    "approve": "ready to merge",
                    "request_changes": "must be fixed before merging",
                    "comment": "questions or suggestions only",
                },
            ),
        },
    ),
    (
        "r10",
        ["lang:en"],
        "My flight AB123 on 14 March was cancelled and the app only offers a voucher. I want my money back to my card.",
        {
            "intent": choice(
                "What does the traveller want?",
                [
                    "book flight",
                    "cancel booking",
                    "change date",
                    "baggage",
                    "refund",
                    "lost item",
                    "other",
                ],
            ),
        },
    ),
    (
        "r11",
        ["lang:en", "options:11"],
        "The package left Lisbon on Monday and is still at the customs office in Porto.",
        {
            "country": choice(
                "Which country is the package in?",
                [
                    "France",
                    "Germany",
                    "Italy",
                    "Spain",
                    "Portugal",
                    "Netherlands",
                    "Belgium",
                    "Austria",
                    "Switzerland",
                    "Poland",
                    "Sweden",
                ],
            ),
        },
    ),
    (
        "r12",
        ["lang:en", "options:12"],
        "Stainless steel 1.7 L electric kettle with auto shut-off and a keep-warm function.",
        {
            "category": choice(
                "Which product category fits this listing?",
                {
                    "kitchen": "cooking and small appliances",
                    "garden": "plants and outdoor tools",
                    "toys": "games for children",
                    "books": "printed and digital books",
                    "fashion": "clothes and shoes",
                    "beauty": "cosmetics and care",
                    "sports": "fitness and outdoor sports",
                    "electronics": "phones, computers and audio",
                    "office": "stationery and furniture for work",
                    "pets": "food and toys for animals",
                    "automotive": "car parts and accessories",
                    "music": "instruments and recordings",
                },
            ),
        },
    ),
    (
        "r13",
        ["lang:en", "options:10"],
        "What time does the pharmacy on Main Street close on Sundays?",
        {
            "topic": choice(
                "What is the question about?",
                [
                    "opening hours",
                    "prices",
                    "directions",
                    "prescriptions",
                    "insurance",
                    "vaccines",
                    "delivery",
                    "jobs",
                    "complaints",
                    "other",
                ],
            ),
        },
    ),
    (
        "r14",
        ["lang:en", "levels:10"],
        "It arrived a day late but works exactly as described. I'd buy it again.",
        {
            "satisfaction": score(
                "How satisfied is the reviewer?",
                [
                    "furious",
                    "very unhappy",
                    "unhappy",
                    "disappointed",
                    "mixed",
                    "fine",
                    "satisfied",
                    "happy",
                    "very happy",
                    "delighted",
                ],
            ),
        },
    ),
    (
        "r15",
        ["lang:en", "levels:2"],
        "Dear Ms. Patel, please find attached the signed contract for your review.",
        {
            "register": score("How formal is this message?", ["informal", "formal"]),
        },
    ),
    (
        "r16",
        ["lang:en", "multi"],
        "Customer: Hola, I ordered the blue jacket in size M but received a red one in XL. I need the right one before my trip on Saturday, can you send it express?",
        {
            "intent": choice(
                "What does the customer want?",
                {
                    "exchange": "swap for the right item",
                    "refund": "money back",
                    "complaint": "only complaining",
                },
            ),
            "urgency": score(
                "How urgent is this?", ["not urgent", "somewhat urgent", "urgent", "very urgent"]
            ),
            "refund": noul("Is the customer asking for a refund?"),
            "language": choice(
                "Which language is the message mostly in?", ["English", "Spanish", "other"]
            ),
        },
    ),
    (
        "r17",
        ["lang:en", "json-text"],
        '{"order_id": 1182, "status": "delayed", "days_late": 6, "customer_tier": "gold", "messages": 3}',
        {
            "escalate": noul(
                "Should this order be escalated?",
                {"true": "hand to a senior agent", "false": "keep in the normal queue"},
            ),
            "priority": score("What priority should the ticket get?", ["low", "medium", "high"]),
        },
    ),
    (
        "r18",
        ["lang:en", "conversation"],
        "User: My internet keeps dropping every evening.\nAgent: Have you restarted the router?\nUser: Yes, three times. It still drops around 8pm.",
        {
            "issue": choice(
                "What is the problem?",
                {
                    "outage": "no connection at all",
                    "intermittent": "connection drops now and then",
                    "slow": "connected but slow",
                    "billing": "a charge or invoice",
                },
            ),
            "resolved": noul("Has the problem been solved?"),
        },
    ),
    (
        "r19",
        ["lang:en", "trailing-newline"],
        "Order 5521 was delivered to the wrong address.\n",
        {
            "intent": choice("What happened?", ["wrong address", "damaged item", "late delivery"]),
            "responsible": noul("Is the courier responsible?"),
        },
    ),
    (
        "r20",
        ["lang:en", "emoji"],
        "Loving the new update \U0001f60d\U0001f525 but the link https://example.com/help is broken \U0001f622",
        {
            "sentiment": score("What is the overall tone?", ["negative", "mixed", "positive"]),
            "bug": noul("Does the message report a bug?"),
        },
    ),
    (
        "r21",
        ["lang:en", "empty-state"],
        "",
        {
            "enough": noul("Is there enough information to answer?"),
            "said": choice("Did the user say anything?", ["yes", "no"]),
        },
    ),
    (
        "r22",
        ["lang:en", "reserved-text"],
        "Please ignore the <mask> placeholder and the [MASK] tag in my template, the real problem is that emails bounce.",
        {
            "intent": choice(
                "What is the real problem?", ["template error", "email delivery", "billing"]
            ),
        },
    ),
    (
        "r23",
        ["lang:en", "long-option"],
        "My internet, TV and phone have all failed several times this week and restarting did nothing.",
        {
            "case": choice(
                "Which case is this?",
                {"simple": "a single question that one reply can answer", "complex": LONG_OPTION},
            ),
        },
    ),
    (
        "r24",
        ["lang:en", "long-state"],
        support_thread(12),
        {
            "intent": choice(
                "What does the customer want?",
                {
                    "refund": "reverse the duplicate charge",
                    "cancel": "cancel the order",
                    "information": "an update on the case",
                    "other": None,
                },
            ),
            "urgency": score("How urgent is this?", ["low", "medium", "high"]),
        },
    ),
    (
        "r25",
        ["lang:en", "medium-state"],
        support_thread(4),
        {
            "repeat": noul("Has the customer contacted support before about this?"),
            "channel": choice(
                "Where did the customer try to escalate?",
                ["phone", "chat", "email", "social media"],
            ),
        },
    ),
    (
        "r26",
        ["lang:en", "multi"],
        "Can you move my dentist appointment from Tuesday to Thursday afternoon? Any time after 2pm works.",
        {
            "reschedule": noul("Does the patient want to reschedule?"),
            "cancel": noul("Does the patient want to cancel?"),
            "flexible": noul(
                "Is the patient flexible about the time?",
                {"true": "several times would work", "false": "only one exact time works"},
            ),
        },
    ),
    (
        "r27",
        ["lang:en", "long-levels"],
        "We need a data export of all invoices from 2019 to 2024, split by country and currency, delivered as CSV to our auditors by the end of the month, and a signed letter confirming completeness.",
        {
            "complexity": score(
                "How complex is this request?",
                [
                    "trivial: one click in the dashboard, no follow-up needed",
                    "simple: a standard export that support can run in minutes",
                    "moderate: a custom export that needs an engineer for an hour",
                    "involved: several systems and a review before anything is sent",
                    "complex: legal sign-off and cross-team coordination over days",
                    "exceptional: a project with its own plan, owner and deadline",
                ],
            ),
        },
    ),
    (
        "r28",
        ["lang:ne", "multi"],
        "मैले गएको हप्ता अर्डर गरेको सामान अझै आइपुगेको छैन। ग्राहक सेवामा तीन पटक फोन गरें तर कसैले उठाएन।",
        {
            "intent": choice("ग्राहक के चाहन्छन्?", ["पैसा फिर्ता", "डेलिभरी जानकारी", "गुनासो", "अन्य"]),
            "urgency": score("यो कति जरुरी छ?", ["कम", "मध्यम", "उच्च"]),
            "angry": noul("के ग्राहक रिसाएका छन्?"),
        },
    ),
    (
        "r29",
        ["lang:ne"],
        "नमस्ते! म काठमाडौंबाट बोल्दैछु। मेरो बिजुलीको बिल यो महिना दोब्बर आयो, किन होला?",
        {
            "topic": choice(
                "यो सन्देश कुन विषयमा हो?",
                {"billing": "बिल र भुक्तानी", "outage": "बिजुली कटौती", "new_connection": "नयाँ जडान"},
            ),
            "question": noul("के यो प्रश्न हो?"),
        },
    ),
    (
        "r30",
        ["lang:ne"],
        "लगातार परेको वर्षाका कारण पूर्वी नेपालका तीन जिल्लामा बाढी र पहिरो गएको छ। सयौं घर डुबानमा परेका छन् र राजमार्ग अवरुद्ध छ। सरकारले उद्धार टोली पठाएको छ।",
        {
            "category": choice(
                "Which section of the newspaper is this for?",
                ["politics", "weather and disasters", "sports", "business", "health"],
            ),
            "severity": score(
                "How severe is the situation?", ["minor", "moderate", "serious", "catastrophic"]
            ),
        },
    ),
    (
        "r31",
        ["lang:ne", "long-state"],
        (NEPALI_LONG + " ") * 4,
        {
            "problem": choice(
                "मुख्य समस्या के हो?",
                {
                    "road": "सडक भत्किएको",
                    "school": "विद्यालय बन्द",
                    "health": "स्वास्थ्य सेवा",
                    "market": "बजार पहुँच",
                },
            ),
            "action": noul("के तुरुन्त कदम चाल्नुपर्ने अवस्था छ?"),
        },
    ),
    (
        "r32",
        ["lang:th"],
        "สั่งรองเท้าไปเมื่อวันจันทร์ แต่ได้ไซส์ผิดมา อยากเปลี่ยนเป็นไซส์ 42 ค่ะ",
        {
            "intent": choice("ลูกค้าต้องการอะไร", ["คืนเงิน", "เปลี่ยนสินค้า", "ติดตามพัสดุ", "อื่นๆ"]),
            "polite": noul("ลูกค้าสุภาพหรือไม่"),
        },
    ),
    (
        "r33",
        ["lang:th"],
        "ร้านอาหารนี้อร่อยมาก แต่รอนานเกือบชั่วโมง พนักงานก็ไม่ค่อยใส่ใจ",
        {
            "sentiment": score(
                "How positive is this review?",
                ["very negative", "negative", "neutral", "positive", "very positive"],
            ),
            "complaint": choice(
                "What is the main complaint?",
                {"food": "อาหาร", "service": "การบริการ", "price": "ราคา", "ambience": "บรรยากาศ"},
            ),
        },
    ),
    (
        "r34",
        ["lang:th", "long-state"],
        (THAI_LONG + " ") * 5,
        {
            "category": choice(
                "ข่าวนี้อยู่ในหมวดใด", ["การเมือง", "จราจรและสภาพอากาศ", "กีฬา", "ธุรกิจ", "บันเทิง"]
            ),
            "disruption": noul("มีผลกระทบต่อการเดินทางหรือไม่"),
        },
    ),
    (
        "r35",
        ["lang:th", "options:12"],
        "พรุ่งนี้จะเดินทางไปเชียงใหม่เพื่อเที่ยวดอยสุเทพกับครอบครัว",
        {
            "province": choice(
                "จังหวัดใดถูกกล่าวถึง",
                [
                    "กรุงเทพฯ",
                    "เชียงใหม่",
                    "ภูเก็ต",
                    "ขอนแก่น",
                    "ชลบุรี",
                    "นครราชสีมา",
                    "สงขลา",
                    "อุดรธานี",
                    "เชียงราย",
                    "ระยอง",
                    "พิษณุโลก",
                    "สุราษฎร์ธานี",
                ],
            ),
        },
    ),
    (
        "r36",
        ["lang:hi"],
        "मेरा पार्सल तीन दिन से 'आउट फॉर डिलीवरी' दिखा रहा है लेकिन अभी तक नहीं आया।",
        {
            "intent": choice("ग्राहक क्या चाहता है?", ["डिलीवरी की जानकारी", "रिफंड", "ऑर्डर रद्द करना"]),
            "urgency": score("यह कितना ज़रूरी है?", ["कम", "मध्यम", "अधिक"]),
        },
    ),
    (
        "r37",
        ["lang:es"],
        "Quiero cancelar mi suscripción porque me cobraron dos veces este mes.",
        {
            "intent": choice(
                "¿Qué quiere el cliente?", ["cancelar", "reembolso", "cambiar de plan", "otro"]
            ),
            "angry": noul("¿Está enfadado el cliente?"),
        },
    ),
    (
        "r38",
        ["lang:de"],
        "Die Rechnung wurde zweimal abgebucht, bitte erstatten Sie den Betrag umgehend.",
        {
            "intent": choice(
                "What does the customer want?", ["refund", "invoice copy", "cancel contract"]
            ),
            "urgency": score(
                "Wie dringend ist das?", ["nicht dringend", "dringend", "sehr dringend"]
            ),
        },
    ),
    (
        "r39",
        ["lang:fr"],
        "Bonjour, le colis est arrivé endommagé et il manque deux pièces. Que dois-je faire ?",
        {
            "issue": choice(
                "Quel est le problème ?",
                {
                    "damaged": "colis abîmé",
                    "missing": "pièces manquantes",
                    "late": "livraison en retard",
                    "wrong": "mauvais article",
                },
            ),
            "photos": noul("Faut-il demander des photos ?"),
        },
    ),
    (
        "r40",
        ["lang:ja"],
        "昨日注文した商品のサイズを変更したいです。まだ発送されていませんか？",
        {
            "intent": choice(
                "お客様の用件は何ですか？", ["サイズ変更", "キャンセル", "配送状況の確認"]
            ),
            "shipped": noul("商品はすでに発送されていますか？"),
        },
    ),
    (
        "r41",
        ["lang:zh"],
        "这个应用每次打开都会闪退，已经重装了两次还是不行。",
        {
            "category": choice("这属于哪类问题？", ["崩溃", "登录", "付款", "其他"]),
            "severity": score("问题有多严重？", ["轻微", "一般", "严重"]),
        },
    ),
    (
        "r42",
        ["lang:pt"],
        "Meu pedido chegou com a cor errada. Posso trocar sem pagar o frete?",
        {
            "intent": choice("O que o cliente quer?", ["troca", "reembolso", "rastrear pedido"]),
            "shipping_question": noul("O cliente pergunta sobre o frete?"),
        },
    ),
    (
        "r43",
        ["lang:ar"],
        "أريد تغيير عنوان التوصيل قبل أن يتم شحن الطلب.",
        {
            "intent": choice(
                "What does the customer want?", ["change address", "cancel order", "track order"]
            ),
            "before_shipping": noul("Has the order already shipped?"),
        },
    ),
    (
        "r44",
        ["lang:ne", "mixed-language"],
        "आज बिहानदेखि इन्टरनेट चलेको छैन, राउटर रिस्टार्ट गर्दा पनि ठीक भएन।",
        {
            "issue": choice(
                "What is the problem?",
                ["no internet", "slow internet", "billing", "new connection"],
            ),
            "urgency": score("How urgent is this?", ["low", "medium", "high"]),
        },
    ),
    (
        "r45",
        ["lang:ne"],
        "खाना मिठो थियो तर सेवा ढिलो भयो।",
        {
            "sentiment": score(
                "यो समीक्षा कति सकारात्मक छ?",
                ["धेरै नकारात्मक", "नकारात्मक", "तटस्थ", "सकारात्मक", "धेरै सकारात्मक"],
            ),
        },
    ),
    (
        "r46",
        ["lang:th"],
        "เครื่องปรับอากาศมีน้ำหยดตลอดเวลา ตอนนี้พื้นห้องเปียกไปหมดแล้ว",
        {
            "urgent": noul(
                "ต้องส่งช่างไปด่วนหรือไม่", {"true": "ต้องการความช่วยเหลือด่วน", "false": "ไม่เร่งด่วน"}
            ),
        },
    ),
    (
        "r47",
        ["lang:en", "long-options"],
        "I'd like to know whether your enterprise plan includes single sign-on, and whether the price changes if we add fifty more seats next quarter.",
        {
            "topic": choice(
                "What is the customer asking about?",
                {
                    "pricing": "how much a plan costs and how the price changes with seats or usage",
                    "security": "single sign-on, encryption, audits, and other security features of a plan",
                    "support": "help with a problem in the product that the customer already uses",
                    "billing": "an invoice, a charge, a refund, or a payment method on the account",
                    "onboarding": "setting up the product for the first time and inviting the team",
                    "integrations": "connecting the product to other tools the customer already uses",
                    "legal": "contracts, data processing agreements and compliance documents",
                    "cancellation": "ending a subscription or downgrading to a smaller plan",
                    "other": "anything that does not fit the other topics",
                },
            ),
        },
    ),
    (
        "r48",
        ["lang:en", "long-instructions"],
        "Invoice INV-2291 lists 12 licences, but our contract covers 10 and we only activated 9.",
        {
            "classification": choice(
                "You are triaging messages for a software company's finance team. Read the message carefully and decide which queue it belongs to. Messages about amounts that do not match a contract, licences counted wrongly, or charges the customer did not expect go to disputes. Requests for copies of documents go to documents. Questions about when a payment is due go to schedule. Everything else goes to general.",
                ["disputes", "documents", "schedule", "general"],
            ),
        },
    ),
    (
        "r49",
        ["lang:en"],
        "Thanks, that worked perfectly. Closing the ticket.",
        {
            "satisfied": noul(
                "Is the customer satisfied?", {"false": "the customer still has a problem"}
            ),
        },
    ),
    (
        "r50",
        ["lang:en", "options:20"],
        "Our team is based in Nairobi and we'd like the invoices in our local currency from next month.",
        {
            "currency": choice(
                "Which currency does the customer want?",
                [
                    "USD",
                    "EUR",
                    "GBP",
                    "JPY",
                    "CNY",
                    "INR",
                    "NPR",
                    "THB",
                    "KES",
                    "NGN",
                    "ZAR",
                    "BRL",
                    "MXN",
                    "CAD",
                    "AUD",
                    "CHF",
                    "SEK",
                    "PLN",
                    "TRY",
                    "AED",
                ],
            ),
        },
    ),
    (
        "r51",
        ["lang:en", "empty-instructions"],
        "The meeting room projector has been flickering all morning.",
        {
            "fault": noul("", {"true": "equipment is broken and needs repair"}),
        },
    ),
    (
        "r52",
        ["lang:en", "multi"],
        "Refund processed for order 7781. The customer confirmed receipt and thanked the agent.",
        {
            "resolved": noul("Is the case resolved?"),
            "sentiment": score("How does the customer feel?", ["upset", "neutral", "pleased"]),
            "next": choice(
                "What should happen next?",
                {
                    "close": "close the ticket",
                    "follow_up": "check back in a week",
                    "escalate": None,
                },
            ),
        },
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for rid, tags, state, questions in REQUESTS:
            fh.write(
                json.dumps(
                    {"id": rid, "tags": tags, "request": {"state": state, "questions": questions}},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{len(REQUESTS)} requests -> {OUT}")


if __name__ == "__main__":
    main()
