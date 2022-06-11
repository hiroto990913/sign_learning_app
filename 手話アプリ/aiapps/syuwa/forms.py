from django import forms

class PhotoForm(forms.Form):
    image = forms.ImageField(widget=forms.FileInput(attrs={'class':'custom-file-input'}))

class AnswerForm(forms.Form):
    subject = forms.CharField(label = '回答（アルファベットで答えてください）', max_length=100)
    
    def __init__(self, *args, **kwargs):
        super(AnswerForm, self).__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"

