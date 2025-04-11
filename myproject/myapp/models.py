from django.db import models

# Create your models here.



class Company(models.Model):
    name = models.CharField(max_length=255)
    contact = models.CharField(max_length=20)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Store hashed passwords if needed
    contact_person = models.CharField(max_length=255, blank=True, null=True)
    address = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name
    


class CompanyDetails(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    file = models.FileField(upload_to='company_files/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Details for {self.company.name}"
