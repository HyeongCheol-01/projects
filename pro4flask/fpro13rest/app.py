from flask import Flask, render_template, request, jsonify # jsonify 추가

# ... (기존 DB 설정 코드 동일) ...
app = Flask(__name__) # 먼저 선언
@app.get('/')
def home():
    return render_template('index.html')

@app.route('/api/friend')
def get_friendFunc():
    name = request.args.get('name',"").strip()
    age_str = request.args.get('age',"").strip()
    # 입력 검증
    if not name:
        return jsonify({"ok":False, "error":"name is required"}), 400
    
    if not age_str.isdigit():
        return jsonify({"ok":False, "error":"age is required"}), 400

    age = int(age_str)
    age_group = f"{(age // 10) * 10}"

    return jsonify({
        "ok":True,
        "name":name,
        "age":age,
        "age_group":age_group,
        "message":f"{name}님은 {age}살 {age_group}입니다"
    })

if __name__=="__main__":
    app.run(debug=True)