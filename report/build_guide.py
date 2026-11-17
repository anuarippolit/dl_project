from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
PRES = ROOT / "presentation"
OUT = PRES / "00_READ_ME_FIRST_project_guide.pdf"

pdfmetrics.registerFont(TTFont("Body", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", "/System/Library/Fonts/Supplemental/Arial Italic.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

INK, ACCENT, SOFT, LINE, WARN, GOOD = (colors.HexColor(c) for c in ["#1f2937", "#2563eb", "#eff6ff", "#cbd5e1", "#fef3c7", "#dcfce7"])
body = ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=15, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#475569"))
cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=13, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Body-Bold")
h1 = ParagraphStyle("h1", parent=body, fontName="Body-Bold", fontSize=18, leading=22, textColor=ACCENT, spaceBefore=8, spaceAfter=8, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Body-Bold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4, keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName="Body-Bold", fontSize=27, leading=32, textColor=ACCENT, spaceAfter=10)
subtitle = ParagraphStyle("subtitle", parent=body, fontSize=13, leading=18, textColor=colors.HexColor("#475569"))


def P(t, st=body):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph("•&nbsp;&nbsp;" + t, ParagraphStyle("b", parent=body, leftIndent=14, firstLineIndent=-10, spaceAfter=3)) for t in items]


def box(t, fill=SOFT):
    x = Table([[Paragraph(t, body)]], colWidths=[16.6 * cm])
    x.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return x


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(st))
    return t


def image(name, width):
    path = PRES / name
    if not path.exists():
        return P(f"(missing figure {name})", small)
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width * cm, height=width * cm * h / w)


story = []

# ------------------------------------------------------------------ cover
story += [
    Spacer(1, 2.6 * cm),
    P("Big model or small model?", title),
    P("Project guide and presentation plan", subtitle),
    Spacer(1, 0.4 * cm),
    P("CSCI 494 Deep Learning. Wav2Vec2 versus spectrogram models on nature sounds and speech: accuracy, cost, and how honest the model's confidence is under noise.", subtitle),
    Spacer(1, 1.0 * cm),
    box("<b>How to use this file.</b> Part 1 explains the whole project from zero, with no jargon. Part 2 is the presentation: 14 slides, who speaks, how long, "
        "what to say, and which picture to put on each slide. Part 3 is a cheat sheet for questions from the audience. Part 4 is the whole presentation plan on one page, with the speakers."),
    Spacer(1, 0.4 * cm),
    box("<b>Pictures.</b> All figures are in this same folder and are named <b>slide_XX_...png</b>. The number in the file name is the number of the slide where it goes. "
        "If a file name has an extra letter (slide_04a, slide_04b), both pictures go on that slide.", GOOD),
    Spacer(1, 0.4 * cm),
    box("<b>Everyone must understand the whole story, not only their own slides.</b> The course says that during the presentation any member can be asked to explain "
        "or modify any part of the code. Read Part 1 completely, then your own slides, then Part 3.", WARN),
    PageBreak(),
]

# ------------------------------------------------------------------ part 1
story += [P("Part 1. The project explained from zero", h1)]
story += [P("1.1 The one-minute version", h2)]
story += [
    P("We want a computer to recognise a short sound: a dog bark, rain, or the spoken word &quot;yes&quot;. We tested four models, from a very old simple one to a huge modern one. "
      "The huge one (Wav2Vec2) was taught on human speech before we ever touched it. So speech is its home ground, and nature sounds are foreign ground."),
    P("We asked two things. <b>First: is the huge model worth its price?</b> It is hundreds of times bigger and slower. <b>Second: when the sound gets noisy, "
      "do the models stay honest about how sure they are?</b> A model that is wrong but says &quot;I am 95% sure&quot; is dangerous, because nobody can see its mistakes."),
    box("<b>The answer in three sentences.</b> (1) The huge model is worth its price only when training data is very small and the sound is the kind it was pre-trained on. "
        "(2) Under strong noise every network, including the huge one, becomes overconfident, and it is much worse on nature sounds than on speech. "
        "(3) Simple fixes help only partly: one correction number does not fix it, and training with noise fixes it for the noise it has seen but costs accuracy on clean sound."),
]

story += [P("1.2 The data", h2)]
story += [
    table(
        [
            ["Dataset", "What it contains", "Size", "For Wav2Vec2 it is..."],
            ["ESC-10", "10 kinds of environmental sound: dog, rain, sea waves, clock tick, sneezing, helicopter, chainsaw, rooster, fire, crying baby. 5-second clips.", "400 clips, 5 folds", "foreign ground (not speech)"],
            ["Speech Commands", "10 spoken words: yes, no, up, down, left, right, on, off, stop, go. 1-second clips.", "tens of thousands of clips", "home ground (speech)"],
            ["Speech Commands, 320 clips (&quot;sc_small&quot;)", "The same words, but we train on only 320 clips. ESC-10 has 240 training clips in each fold, so the sizes are close.", "320 for training", "home ground, little data"],
        ],
        [3.3, 7.4, 2.9, 3.0],
    ),
    Spacer(1, 5),
    P("<b>Why the third dataset?</b> ESC-10 is small and Speech Commands is big. If we compare them directly, we cannot tell whether a difference comes from the type of sound or from the amount of data. "
      "The 320-clip version almost removes the data-size difference (it is close to the 240 training clips ESC-10 has in each fold) and leaves mostly the type of sound."),
    P("<b>How the data is split.</b> ESC-10 uses its 5 official folds: in each round one fold is the test set, the next fold is for validation, and the remaining three train the model. "
      "Results of the five rounds are put together. Speech Commands uses the official validation and test lists."),
]

story += [P("1.3 A sound is a picture", h2)]
story += [
    P("A computer cannot look at a wave of air pressure the way we hear it. A common trick is to turn the sound into a picture called a <b>spectrogram</b>. "
      "Time goes left to right, pitch goes bottom to top, and brightness shows how strong that pitch is at that moment. Once the sound is a picture, we can use the same kind of networks that recognise photos."),
    P("Wav2Vec2 does it differently: it reads the raw sound wave directly and learns its own features."),
    Spacer(1, 3),
    KeepTogether([image("slide_02_sound_as_picture.png", 13.5), P("Figure: the same sounds as a wave (top) and as a spectrogram (bottom). File: slide_02_sound_as_picture.png", small)]),
]

story += [P("1.4 The four models", h2)]
story += [
    table(
        [
            ["Model", "What it is, in plain words", "Size", "Training time on full Speech Commands"],
            ["Random Forest", "The old-school baseline. A crowd of decision trees looking at hand-made sound summaries (MFCC). No neural network.", "no network", "about 40 seconds"],
            ["CNN", "A small convolutional network, trained from nothing on the spectrogram pictures.", "0.24 million parameters", "8 minutes"],
            ["ResNet-18", "A well-known network for photos, already taught on millions of ImageNet photos, then fine-tuned on our spectrograms.", "11.2 million", "16 minutes"],
            ["Wav2Vec2", "A huge network pre-trained on hours of human speech, fine-tuned on the raw sound wave.", "94.4 million", "94 minutes"],
        ],
        [2.7, 7.5, 3.1, 3.3],
    ),
    Spacer(1, 4),
    P("<b>Pre-trained</b> means the model already learned something general before our task, like a person who can already read before taking our class. Wav2Vec2 learned from speech, ResNet from photos."),
    P("Training recipe (useful if someone asks): AdamW optimiser, cosine learning-rate schedule, dropout 0.3 on the classification layer, noise-free training, SpecAugment-style masks on the spectrogram for CNN and ResNet. "
      "The best epoch is chosen by the lowest validation loss. Wav2Vec2 is fully fine-tuned except for its first convolutional feature layers, which are frozen."),
]

story += [P("1.5 Noise and what &quot;honest&quot; means", h2)]
story += [
    P("<b>Noise.</b> We train every model on clean sound only. At test time we add noise to the test clips. The strength of noise is measured by SNR in decibels (dB): "
      "20 dB is weak noise, 10 dB is medium, and <b>0 dB means the noise is as loud as the sound itself</b>. Three kinds of noise:"),
]
story += bullets([
    "<b>white noise</b>: pure hiss;",
    "<b>speech-data background</b>: real background recordings from the Speech Commands dataset (dishes, a running tap, an exercise bike);",
    "<b>nature-sound noise</b>: random pieces of other nature sounds from ESC-50 (wind, engines, tools), none of which are in ESC-10.",
])
story += [
    P("<b>Confidence and honesty.</b> A model gives an answer and also a confidence, for example &quot;dog, 95% sure&quot;. Think of a weather forecaster. If on all days when he says &quot;90% rain&quot; "
      "it rains on about 90% of them, he is honest. If it rains on only 40%, he is overconfident. We measure this with <b>ECE</b> (expected calibration error): 0 is perfectly honest, bigger is worse. "
      "We also look at <b>overconfidence = average confidence minus accuracy</b>: above zero means the model brags, below zero means it is too modest."),
    Spacer(1, 3),
    KeepTogether([image("slide_04b_honest_confidence.png", 13.5), P("Figure: ResNet-18 on ESC-10. With clean sound the bars follow the dashed line (honest). With strong noise it says 90% sure and is right about 15% of the time. File: slide_04b_honest_confidence.png", small)]),
    box("<b>Why it matters.</b> If noise breaks a model but it shows low confidence, a person can still use it carefully. If it breaks and stays confident, the mistakes are invisible."),
    PageBreak(),
]

story += [P("1.6 The eight questions and their answers", h1)]
story += [
    P("Our presentation is built from questions. Each question is a slide with a short answer. Some questions come from the first full run of all models, others come from extra experiments we added to check ourselves. "
      "The names Q1 to Q8 below are the <b>slide question numbers</b>. In the code folder the notebooks have different numbers, see the table at the end of this section.", small),
]
QA = [
    ("Q1. Is the huge model worth its price?",
     "Only sometimes. On nature sounds it is not better than ResNet-18 (0.82 to 0.85 against 0.87 clean accuracy in two runs). On full speech all neural networks are almost equal (0.97 to 0.98) while Wav2Vec2 is 390 times bigger than the CNN and trains 11 times longer. "
     "It clearly wins only with 320 training clips of speech (0.95 against 0.60 for the CNN)."),
    ("Q2. Are the differences between models real, or just luck?",
     "We repeated cheap experiments with 3 random seeds. On ESC-10, ResNet (0.877) is reliably better than CNN (0.834) and Random Forest (0.789). "
     "A surprise: on 320 clips, Wav2Vec2 completely failed on one seed out of three (accuracy 0.10, it predicted one class for everything), while the other two seeds gave 0.94 and 0.95. Fine-tuning a huge model on tiny data is unstable."),
    ("Q3. Is Wav2Vec2 really better with little data, or were the small models just under-trained?",
     "Partly under-trained. With only 320 clips, the CNN and ResNet got about 150 to 200 training steps. When we trained them 7 to 10 times longer, the CNN rose from 0.60 to 0.79 and ResNet from 0.52 to 0.66. "
     "Wav2Vec2 stayed far ahead (0.96), so the claim survives, but the gap is about 17 points, not 35."),
    ("Q4. Do models become overconfident under noise, and is it worse on nature sounds than on speech?",
     "Yes. At 0 dB white noise on ESC-10, every neural network, including Wav2Vec2, is almost at chance (11 to 13% accuracy) while confidence stays high: overconfidence is about +0.7 to +0.8. "
     "On Speech Commands the same networks stay honest (overconfidence between -0.03 and +0.05). So the huge model is not special: on foreign ground all models brag."),
    ("Q5. Is that because of the type of sound, or simply because the model is broken?",
     "To check, we compare models at the same accuracy instead of the same noise level. When accuracy is below 20%, ESC-10 models are overconfident by +0.77 and the 320-clip speech models by +0.24; "
     "between 20% and 40% it is +0.41 against +0.10. So the type of sound matters, not only the breakage. Caution: only a few points per box and full Speech Commands never gets that low."),
    ("Q6. Does the result depend on which noise we used?",
     "Yes, a lot. On ESC-10 at 0 dB, accuracy is 0.11 to 0.13 with white noise, 0.24 to 0.25 with speech-data background, and 0.51 to 0.56 with nature-sound noise. "
     "Overconfidence drops from about +0.75 to between -0.01 and +0.21 on the nature-sound noise. The dramatic result is partly a property of our noise, and we must say so."),
    ("Q7. Can a simple correction (temperature scaling) fix overconfidence?",
     "Not with one number. Temperature scaling divides the model's raw scores by one number to make confidence more modest. Fitted on clean sound it does not help under noise. "
     "Fitted on all noise levels at once it helps on known noise (ESC-10: ECE 0.28 to 0.21) but ruins clean sound (0.12 to 0.32) and does not transfer to a new noise kind. "
     "If we knew the noise in advance, the best case is excellent (ECE 0.07), so the problem is fixable only with knowledge we normally do not have."),
    ("Q8. Can we fix it by training with noise?",
     "On the noise it has seen, very well: on ESC-10, ResNet ECE falls from 0.34 to 0.04 and Wav2Vec2 from 0.33 to 0.07. On a new kind of noise the gain is smaller and not for everyone. "
     "The price: clean accuracy drops by 4 to 12 points on ESC-10 (Wav2Vec2 0.85 to 0.73), and with only 320 clips training with noise hurts badly (Wav2Vec2 0.94 to 0.61)."),
]
for q, a in QA:
    story += [KeepTogether([P(q, h2), P(a)])]

story += [
    P("Where each question comes from in the code", h2),
    table(
        [
            ["Slide question", "Source in the project"],
            ["Q1, Q4", "The first full run: results/results.csv, cost.csv, reliability.csv, notebooks/results.ipynb"],
            ["Q2", "notebooks/q1_seeds.ipynb"],
            ["Q3", "notebooks/q3_undertraining.ipynb"],
            ["Q5", "notebooks/q5_confidence_vs_accuracy.ipynb"],
            ["Q6", "notebooks/q6_noise_source.ipynb"],
            ["Q7", "notebooks/q7_temperature.ipynb"],
            ["Q8", "notebooks/q8_noise_training.ipynb"],
        ],
        [4.0, 12.6],
    ),
    PageBreak(),
]

# ------------------------------------------------------------------ part 2
SPEAKERS = [
    ["1 to 2", "A", "Problem and goal, sound as a picture", "about 1 min 15 s"],
    ["3 to 4", "B", "The models, noise and honesty", "about 1 min 40 s"],
    ["5 to 7", "C", "Is it worth the price? Are gaps real? Under-training", "about 2 min 20 s"],
    ["8 to 9", "D", "Overconfidence under noise, and is it the domain?", "about 2 min 20 s"],
    ["10 to 12", "E", "Noise source, temperature scaling, training with noise", "about 2 min 20 s"],
    ["13 to 14", "F", "Conclusions and limitations", "about 1 min 15 s"],
]
story += [P("Part 2. The presentation plan", h1)]
story += [
    P("Target length: 10 to 12 minutes. The plan below adds up to about 11 minutes. If you run late, drop slide 6 and slide 10 into the backup section (see their notes)."),
    P("Template for every question slide: <b>the question as the title, the answer in one big line, one picture, one line of &quot;why it matters&quot;.</b> Keep text on slides very short and put the explanation in your voice."),
    table([["Slides", "Speaker", "Topic", "Time"]] + SPEAKERS, [2.0, 2.0, 9.4, 3.2]),
    Spacer(1, 4),
    P("Suggestion: the person who knows the code best takes D, because slides 8 and 9 are the core of the project.", small),
]

SLIDES = [
    dict(n=1, t="Is a huge pre-trained model worth it, and can we trust it in noise?", who="A", sec=35, fig=None,
         key="We compare a huge model trained on speech with small models, and ask whether the big one is worth its price and whether it is honest when sound gets noisy.",
         onslide="Title. Two bullets: (1) Is the huge model worth its price? (2) Does it stay honest under noise? One line at the bottom: team, course.",
         say="When you want to recognise a sound, you can use a small model or a huge pre-trained one. The huge one is hundreds of times bigger and slower. Our project asks two things. "
             "First, is that size worth it? Second, when the sound is noisy, does the model know when it is wrong? A model that is wrong and sure of itself is dangerous.",
         ask="Why this topic? Because pre-trained models are popular and expensive, and people rarely check whether their confidence is honest."),
    dict(n=2, t="A sound can be turned into a picture", who="A", sec=40, fig=["slide_02_sound_as_picture.png"],
         key="Sound becomes a spectrogram, so image networks can read it. We use two datasets: nature sounds and speech.",
         onslide="The figure. Under it, two small boxes: ESC-10 = 10 nature sounds, 400 clips. Speech Commands = 10 words. One note: 320 clips version for fair comparison.",
         say="Top is the raw sound wave of a dog barking and of the word yes. Bottom is the same sound as a picture: time left to right, pitch bottom to top, brightness is strength. "
             "Now image networks can read sound. We use two datasets. ESC-10 has nature sounds, which is foreign ground for a model taught on speech. Speech Commands has spoken words, which is home ground. "
             "We also made a small version with only 320 clips, so both datasets have a similar size (240 training clips per fold in ESC-10, 320 here).",
         ask="What is a spectrogram? A picture of how strong each pitch is over time."),
    dict(n=3, t="Four models, from tiny to huge", who="B", sec=45, fig=["slide_03_models.png"],
         key="Random Forest is the baseline, CNN is small, ResNet-18 is medium and pre-trained on photos, Wav2Vec2 is huge and pre-trained on speech.",
         onslide="The figure only. Add the word BASELINE next to Random Forest and PRE-TRAINED next to ResNet-18 and Wav2Vec2.",
         say="We compare four models. Random Forest is the old baseline with no neural network. The CNN is a small network trained from scratch on the spectrogram pictures. "
             "ResNet-18 is a known image network that already saw millions of photos, and we fine-tune it on spectrograms. Wav2Vec2 is huge: 94 million parameters, 390 times more than the CNN. It reads the raw wave and was pre-trained on speech.",
         ask="Why these models? One of each kind: no network, small network, pre-trained on images, pre-trained on speech."),
    dict(n=4, t="Adding noise, and what honest confidence means", who="B", sec=55, fig=["slide_04a_noise_levels.png", "slide_04b_honest_confidence.png"],
         key="We add noise at test time only. 0 dB means noise as loud as the sound. A model is honest when its confidence matches how often it is right.",
         onslide="Left: slide_04a (noise levels). Right: slide_04b (honest confidence). Caption: confidence 90% should mean right 90% of the time.",
         say="We train on clean sound and add noise only when testing. At 0 dB the noise is as loud as the word itself. Now the second idea: honesty. A model says how sure it is. "
             "Look at the right picture: with clean sound the bars follow the dashed line, so when it says 90% it is right about 90% of the time. With strong noise it still says 90% but is right only about 15% of the time. That is what we call overconfidence, and we measure it with a number called ECE.",
         ask="Why train on clean sound? To test how a model behaves when conditions change after training, which is how real deployment works."),
    dict(n=5, t="Q1. Is the huge model worth its price?", who="C", sec=50, fig=["slide_05_accuracy_vs_cost.png"],
         key="Not on nature sounds, and almost not on full speech. It pays off only with very little speech data.",
         onslide="Answer line: NOT ALWAYS. The figure. One line: 390x more parameters, 11x training time for +0.01 on full speech.",
         say="Each dot is a model: further right is more training time, higher is more accurate. On nature sounds, on the left, the huge Wav2Vec2 is not better than ResNet-18 and is much slower. "
             "On full speech, on the right, all networks are within a point or two of each other, while Wav2Vec2 takes 94 minutes against 8 for the small CNN. Its only clear win is with 320 speech clips, which we will check next.",
         ask="Numbers: ESC-10 clean accuracy RF 0.78, CNN 0.83, ResNet 0.87, Wav2Vec2 0.82 to 0.85. Speech full: 0.63, 0.97, 0.98, 0.98."),
    dict(n=6, t="Q2. Are the differences real or luck? (backup if short on time)", who="C", sec=40, fig=["slide_06_seeds.png"],
         key="We repeated runs 3 times. ResNet is reliably better than CNN on ESC-10. Wav2Vec2 on 320 clips failed completely in one run out of three.",
         onslide="Answer line: MOSTLY REAL, BUT WAV2VEC2 IS UNSTABLE ON TINY DATA. The figure.",
         say="One run can be luck, so we repeated the experiment three times with different random seeds. Each dot is one run. On nature sounds the dots sit tightly together, so ResNet really beats the CNN. "
             "On the right, with 320 speech clips, two Wav2Vec2 runs reach 0.95 but one run completely failed and predicted a single class. Fine-tuning a huge model on tiny data is not only expensive but unstable.",
         ask="What is a seed? The number that fixes the randomness of the starting weights and the data order."),
    dict(n=7, t="Q3. Were the small models just under-trained?", who="C", sec=50, fig=["slide_07_undertraining.png"],
         key="Partly yes. Training the small models 7 to 10 times longer closed about half of the gap, but Wav2Vec2 stays ahead.",
         onslide="Answer line: PARTLY. GAP FELL FROM 35 TO 17 POINTS. The figure.",
         say="On 320 clips the small models only got about 150 to 200 training steps, so maybe they just did not finish learning. We trained them much longer. The CNN went from 0.60 to 0.79 and ResNet from 0.52 to 0.66. "
             "So half of the advantage of Wav2Vec2 on small data was just under-training. But Wav2Vec2 still scores 0.96, so pre-training on matching speech really helps when data is scarce.",
         ask="Why not train even longer? The curves flatten after about 600 steps."),
    dict(n=8, t="Q4. Do models become overconfident under noise?", who="D", sec=90, fig=["slide_08_overconfidence_noise.png"],
         key="Yes on nature sounds, for every neural network including Wav2Vec2. On speech they stay honest.",
         onslide="Answer line: ON NATURE SOUNDS, EVERY NETWORK BRAGS. The figure. This is the core slide, give it the most time.",
         say="This is the main result. The vertical axis is overconfidence: how much the model's confidence is higher than its accuracy. Zero means honest. On the left, nature sounds: as noise grows, all three neural networks rise to about +0.7 or +0.8. "
             "At 0 dB their accuracy is near chance, around 12%, but they are still confident. Wav2Vec2, the model taught on speech, behaves the same as the small ones, so it is not special. On the right, speech: the same networks stay near zero. "
             "So the type of sound strongly changes how trustworthy a model is under noise.",
         ask="Is Wav2Vec2 more honest on speech? About equally honest as the others (overconfidence +0.01). Random Forest is the opposite: it is modest, not boastful."),
    dict(n=9, t="Q5. Is it the type of sound, or just a broken model?", who="D", sec=50, fig=["slide_09_same_accuracy.png"],
         key="At the same accuracy, nature sounds are more overconfident than speech, so the type of sound matters.",
         onslide="Answer line: THE TYPE OF SOUND MATTERS, NOT ONLY THE BREAKAGE. The figure.",
         say="A fair doubt: at 0 dB models are broken, so of course they are overconfident. To check, we compare models at the same accuracy. Each dot is one model in one noise condition. Look at accuracy 0.2 to 0.4: "
             "the red dots, nature sounds, are far above the green dots, small-data speech. So at equal breakage, nature sounds are still more overconfident. Caution: there are few dots, and full speech never gets that low.",
         ask="Why no blue dots on the left? Full Speech Commands stays above 60% accuracy even in strong noise."),
    dict(n=10, t="Q6. Does the kind of noise matter? (backup if short on time)", who="E", sec=40, fig=["slide_10_noise_source.png"],
         key="Yes. With noise made of other nature sounds, models stay far more accurate and far less overconfident.",
         onslide="Answer line: YES, A LOT. Say the dramatic result depends partly on our noise. The figure.",
         say="Is our result only about the noise we picked? We tried three kinds at 0 dB on ESC-10: white hiss, background recordings from the speech dataset, and pieces of other nature sounds. "
             "With hiss, accuracy is about 12%; with the nature-sound noise it is above 50%, and overconfidence almost disappears. So the result is real but depends on the noise, and we say so honestly.",
         ask="Why the speech background noise at all? It was our first choice in the main run, taken from the speech dataset."),
    dict(n=11, t="Q7. Can a simple correction fix overconfidence?", who="E", sec=45, fig=["slide_11_temperature.png"],
         key="Not with one number. It helps on known noise but ruins clean sound and does not carry over to a new kind of noise.",
         onslide="Answer line: NOT WITH ONE NUMBER. The figure. Legend in the figure explains the four bars.",
         say="Temperature scaling is a cheap trick: one number divides the model's scores to make it more modest. Fitted on clean sound, it does not help under noise (blue). Fitted on all noise levels together (orange), it helps on noise it saw, "
             "but makes clean sound much worse and does not help on a new kind of noise. Only if we know the noise in advance (green) is it excellent, and in practice we do not.",
         ask="What is ECE again? Average gap between confidence and accuracy over confidence bins. Lower is better."),
    dict(n=12, t="Q8. Can we fix it by training with noise?", who="E", sec=55, fig=["slide_12_noise_training.png", "slide_12b_small_data_warning.png"],
         key="Yes for the noise the model has seen, with a price on clean sound, and it hurts when data is tiny.",
         onslide="Answer line: YES FOR KNOWN NOISE, BUT IT HAS A PRICE. The main figure. Show slide_12b as a small inset or in backup.",
         say="Instead of fixing confidence afterwards, we add noise during training. Pale bars are the clean-trained models, solid bars are trained with noise. In the middle, dishonesty on known noise collapses: ResNet from 0.34 to 0.04, Wav2Vec2 from 0.33 to 0.07. "
             "On a new kind of noise on the right the gain is smaller and not for everyone. On the left you see the price: clean accuracy drops, for Wav2Vec2 from 0.85 to 0.73. And with only 320 clips training with noise hurts a lot.",
         ask="Why is the new noise result mixed? The model only learned the noise it practised on."),
    dict(n=13, t="Conclusions", who="F", sec=45, fig=None,
         key="Big is not automatically better, trust depends on the sound type and the noise, and cheap fixes are partial.",
         onslide="Four short lines, see the say box. No picture.",
         say="One: a huge pre-trained model pays off only with very little data and matching sound; otherwise a small model is about as good. Two: under strong noise, every network becomes overconfident on nature sounds, and Wav2Vec2 is no exception. "
             "Three: it depends on the type of sound and on the noise. Four: one correction number does not fix it, and training with noise fixes known noise at the cost of clean accuracy.",
         ask="What would you tell a practitioner? Test your model under noise, and do not trust its confidence without checking."),
    dict(n=14, t="Limitations and what is next", who="F", sec=30, fig=None,
         key="We were careful, but the study is small.",
         onslide="Four short bullets: one run for most extra experiments; ESC-10 has only 400 test clips; datasets differ in more than the type of sound; synthetic noise.",
         say="Most extra experiments used one seed, and ESC-10 has only 400 test clips, so differences of two or three points are not reliable. ESC-10 and speech differ in more than the type of sound, so our result agrees with the idea but does not prove it. "
             "Noise is added by computer. Next steps would be more seeds, real recordings in noisy places, and other pre-trained models.",
         ask="What did not work? Temperature scaling with one number, and training with noise on 320 clips."),
]

for s in SLIDES:
    head = P(f"Slide {s['n']}. {s['t']}", h2)
    meta = P(f"Speaker <b>{s['who']}</b>, about <b>{s['sec']} s</b>. Figure file(s): <b>{', '.join(s['fig']) if s['fig'] else 'none (text only)'}</b>", small)
    rows = [
        ["Key message", s["key"]],
        ["On the slide", s["onslide"]],
        ["What to say", s["say"]],
        ["If asked", s["ask"]],
    ]
    t = table(rows, [2.6, 14.0], header=False)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), SOFT), ("FONTNAME", (0, 0), (0, -1), "Body-Bold")]))
    block = [head, meta, t]
    if s["fig"]:
        block += [Spacer(1, 4), image(s["fig"][0], 9.0 if len(s["fig"]) == 1 else 8.0)]
        if len(s["fig"]) > 1:
            block += [Spacer(1, 3), image(s["fig"][1], 8.0)]
    story += [KeepTogether(block), Spacer(1, 6)]

story += [PageBreak()]

# ------------------------------------------------------------------ part 3
story += [P("Part 3. Cheat sheet for questions from the audience", h1)]
CHEAT = [
    ("Why train on clean sound only?", "We wanted to test how models cope when conditions change after training. This is the usual deployment situation. In the extra experiment Q8 we also trained with noise."),
    ("What is ECE?", "Expected calibration error. We split predictions into 10 bins by confidence, and in each bin compare average confidence with real accuracy. ECE is the weighted average of those gaps. 0 is perfect."),
    ("What is temperature scaling?", "After training, divide the model's raw scores by one number T. T above 1 makes the model more modest, T below 1 more confident. It does not change the answer, only the confidence. T is chosen to minimise the log-loss on validation data."),
    ("What does 0 dB mean?", "The noise has the same power as the sound. We compute signal power only over the loud parts of the clip, so silence does not make the noise look louder than it is."),
    ("Why is ESC-10 so small?", "It is a standard small benchmark of 400 clips. That is also why we made a 320-clip version of speech: a similar size, but a different type of sound."),
    ("Why one seed for many experiments?", "Time. Wav2Vec2 on ESC-10 takes about an hour per run. We repeated the cheap ones with 3 seeds (Q2). We say openly that most results are single-seed."),
    ("Is the sound domain really the cause?", "We do not prove it. Nature sounds and speech differ in clip length, classes and noise source too. Our results agree with the idea (Q4, Q5) but a controlled proof would need more datasets."),
    ("Why did Wav2Vec2 fail in one run?", "Fine-tuning a huge model on 320 clips is known to be unstable. In that run the model collapsed to predicting one class. We kept it in the results on purpose."),
    ("Did you freeze Wav2Vec2?", "Only its first convolutional feature layers. The transformer layers and the head are fine-tuned. The original proposal said frozen; we changed it to get better accuracy."),
    ("How did you pick the best epoch?", "The epoch with the lowest validation loss on clean validation data."),
    ("What was the CNN architecture?", "Four convolution blocks (32, 64, 128, 128 channels) with batch norm, ReLU and max pooling, then global average pooling and one linear layer. 0.24 million parameters."),
    ("What exactly goes into the spectrogram?", "A log-mel spectrogram with 64 mel bands, window 1024, hop 160, sound resampled to 16 kHz. ResNet gets it copied into 3 channels."),
    ("Is the noise the same for every model?", "Yes. The noise generator uses a fixed seed, so every model sees exactly the same noisy clips."),
    ("Why is Random Forest underconfident?", "A forest averages many trees, so its probabilities are cautious by nature. It is modest in noise, not boastful."),
    ("What would you do next?", "More seeds, real noisy recordings, more datasets, and other pre-trained models such as HuBERT or AST."),
]
story += [table([["Question", "Short answer"]] + [[q, a] for q, a in CHEAT], [4.6, 12.0])]

# ------------------------------------------------------------------ part 4
story += [PageBreak(), P("Part 4. Presentation plan at a glance", h1)]
story += [
    P("Target length 10 to 12 minutes. Template for every question slide: <b>the question as the title, the answer in one big line, one picture, one line of why it matters.</b>"),
    P("Who speaks", h2),
    table([["Slides", "Speaker", "Topic", "Time"]] + SPEAKERS, [2.0, 2.0, 9.4, 3.2]),
    Spacer(1, 4),
    P("Suggestion: the person who knows the code best takes D, because slides 8 and 9 are the core of the project.", small),
    P("All slides", h2),
    table(
        [["#", "Slide", "Who", "Sec", "Figure file (same folder)"]]
        + [[str(x["n"]), x["t"], x["who"], str(x["sec"]), ", ".join(x["fig"]) if x["fig"] else "none, text only"] for x in SLIDES]
        + [["", "<b>Total</b>", "", f"<b>{sum(x['sec'] for x in SLIDES)} (about {sum(x['sec'] for x in SLIDES) / 60:.0f} min)</b>", ""]],
        [0.8, 6.0, 1.3, 2.2, 6.3],
    ),
    Spacer(1, 6),
    box("<b>If you run late:</b> move slide 6 (are the gaps real) and slide 10 (does the noise kind matter) to the backup section and mention them in one sentence."),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.drawString(2 * cm, 1.2 * cm, "CSCI 494: project guide and presentation plan")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm, title="Big model or small model? Project guide and presentation plan", author="")
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("saved", OUT)
