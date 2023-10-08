import DLmodel
import numpy as np
from flask import Flask, request, \
    render_template
app = Flask(__name__)
import pickle

model = pickle.load(open('model.pkl','rb'))

@app.route("/",methods=['GET'])
def main():
    return render_template("index.html")

@app.route("/predict", methods=['post'])
def pred():
    features = [request.data]
    x=DLmodel.preprocess(features)
    x=DLmodel.label(x)
    x=DLmodel.pad(30,x)
    pred = model.predict(x)
    output=np.argmax(pred)
    ans=DLmodel.out_val_mapp[output]
    #print(DLmodel.out_val_mapp,pred,output)
    #result = json.dumps(data) 
    print (ans)
    return jsonify({'prediction': ans})
    
if __name__=='__main__':
    
    app.run(host='127.0.0.1',port=5000)
    
    
    
    
    
    
    
    
    