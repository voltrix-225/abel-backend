import torch
from abel import BigramLanguageModel, encode, decode
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = BigramLanguageModel()
model.load_state_dict(torch.load('model/ABEL_.pth'))
model.to('cpu')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods = ['POST'])
def generate_lyrics():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error" : "Enter a prompt"}), 400
    

    seed_txt = "Verse: \n" + prompt
    seed_ids = torch.tensor([encode(seed_txt)], dtype = torch.long)
    output = model.generate(seed_ids, max_new_tokens=1200)
    lyrics = decode(output[0].tolist())

    return jsonify({"lyrics" : lyrics}) 

if __name__ == '__main__':
    app.run(debug = True)