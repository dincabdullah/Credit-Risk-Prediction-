from django.db import models

# Create your models here.
class Person(models.Model):
    age = models.IntegerField()
    sex = models.BooleanField()
    job = models.IntegerField()
    housing = models.CharField(max_length = 50)
    savingaccounts = models.CharField(max_length = 50, null = True)
    checkingaccount = models.CharField(max_length = 50, null = True)
    creditamount = models.IntegerField()
    duration = models.IntegerField()
    purpose = models.CharField(max_length = 50)
    risk = models.CharField(max_length = 50)

