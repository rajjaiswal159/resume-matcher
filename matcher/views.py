from django.shortcuts import render
from .services.nlp_engine import final_similarity
from .services.pdf_parser import extract_text_from_pdf

def home(request):
    result = None
    error = None

    if request.method == "POST":
        jd = request.POST.get("jd")
        resume_text = ""

        if request.FILES.get("resume_pdf"):
            pdf_file = request.FILES["resume_pdf"]

            if not pdf_file.name.endswith('.pdf'):
                return render(request, "index.html", {"error": "Only PDF allowed"})

            resume_text = extract_text_from_pdf(pdf_file)

        else:
            resume_text = request.POST.get("resume")

        if not resume_text or not jd:
            return render(request, "index.html", {
                "error": "Missing input"
            })

        result = final_similarity(resume_text, jd)

    return render(request, "index.html", {
        "result": result,
        "error": error
    })