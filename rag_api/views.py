from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from src.RAG import RAG


rag_system = RAG(
    model_name="sberbank-ai/sbert_large_nlu_ru",
    json_data_path="data/json",
    index_file_name="index",
    vector_store_path="data/vector_store",
    gigachat_token="YjllY2FhYjgtNGRlMC00MDA4LWIwZmYtNjdlNjY0ZmI5OTc4OmI5YTY3NjYwLWJkMmQtNDNmZi04YzViLTU2MWMxYTE0MjFlMw==",
    llm_model="GigaChat-2-Max",
    retrieval_k=10
)

@csrf_exempt
def ask_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    
    try:
        body = json.loads(request.body.decode("utf-8"))
        question = body.get("question")

        if not question:
            return JsonResponse({"error": "Field 'question' is required"}, status=400)

        result = rag_system.ask(question)
        return JsonResponse(result, status=200, json_dumps_params={"ensure_ascii": False})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
