from re import template
from PIL import Image
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import AnswerForm, PhotoForm
from .models import Photo, Predict, Question
import io, base64
import unicodedata


def index(request):
    template = loader.get_template('syuwa/index.html')
    context = {'form': PhotoForm()}
    return HttpResponse(template.render(context, request))

def gazou(request):
    list = []
    if not request.method == 'POST':
        return
        redirect('syuwa:index')

    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Formが不正です')
        
    photo = Photo(image=form.cleaned_data['image'])
    photo.predict()
    template = loader.get_template('syuwa/answer.html')

    context = {
        'form' : AnswerForm(),
        'photo_data': photo.image_src(),
    }

    return HttpResponse(template.render(context, request))



def answer(request):
    predicted = Predict().predict1()
    
    template = loader.get_template('syuwa/result.html')
    params = {
        'predicted': predicted,
        'answer':'',
        'form':None,
        'hyouzi' :''
    }
    if request.method == 'POST':
        form = AnswerForm(request.POST)
        answer = request.POST["subject"].upper()
        answer = unicodedata.normalize("NFKC", answer)
        
        params['answer'] = answer
        params['form'] = form
        if params['answer'] == params['predicted']:
            params['hyouzi'] = '正解です！引き続き頑張っていきましょう！'
        elif params['answer'] != params['predicted']:
            params['hyouzi'] = '不正解です！もう一度頑張りましょう！'

    else:
        params['form'] = AnswerForm()
    
    return HttpResponse(template.render(params, request))

