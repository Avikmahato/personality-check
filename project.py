from flask import Flask,render_template,request
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
with open("model.pkl","rb") as file:
    model=pickle.load(file)
sc=StandardScaler()
app=Flask(__name__)

@app.route("/")

def start():
    return render_template('index.html')
@app.route("/",methods=["POST"])
def getData():
    values=[int(x) for x in request.form.values()]
    arr_values=[np.array(values)]
    result=model.predict(arr_values)
    return render_template("index.html",data=result)
    
if __name__=="__main__":
    app.run(debug=True)