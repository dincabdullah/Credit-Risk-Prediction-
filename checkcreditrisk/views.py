from django.shortcuts import render
from static.preprocessing.prep import execute_order_by


# Create your views here.
def index(request):
    if request.method == "GET":
        return render(request, "checkcreditrisk/index.html")
    else:
        result = execute_order_by(request)
        credit_risk_result = ""
        if result == 0 or result == "0":
            credit_risk_result = "Good"
        else:
            credit_risk_result = "Bad"

        return render(request, "checkcreditrisk/index.html", {
            'isOk' : True,
            'risk' : credit_risk_result
        })