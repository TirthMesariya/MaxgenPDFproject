"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . views import *
from myapp import views
from myapp.views import PredictAPIView,UploadZipView
urlpatterns = [
    path('api/', PredictAPIView.as_view()),
    path('add_company', views.add_company, name='add_company'), 
    path('company_list', views.company_list, name='company_list'),
    path('add_CompanyDetails', views.add_CompanyDetails, name='add_CompanyDetails'), 
    path('add_list', views.add_list, name='add_list'), 
    path('company_detail123/<int:detail_id>/', views.company_detail123, name='company_detail123'), 
    path('', views.index, name='index'),
    path('UploadZipView', UploadZipView.as_view(), name='UploadZipView'),
    path('zip_folder_data_show', UploadZipView.as_view(), name='zip_folder_data_show'),
     


]
