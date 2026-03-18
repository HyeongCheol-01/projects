// 각종 요소
const code = document.querySelector("#code");
const sang = document.querySelector("#sang");
const su = document.querySelector("#su");
const dan = document.querySelector("#dan");

const msg = document.querySelector("#msg");
const tbody = document.querySelector("#tbody");

const btnAdd = document.querySelector("#btnAdd");
const btnUpdate = document.querySelector("#btnUpdate");
const btnDelete = document.querySelector("#btnDelete");
const btnReload = document.querySelector("#btnReload");

function setMsg(text){
    msg.textContent = text;
}

// 입력 폼 초기화
function clearForm(){
    code.value = "";
    sang.value = "";
    su.value = "";
    dan.value = "";
}

// 전체 자료 읽기
async function loadAll(){
    const res = await fetch("/api/sangdata");
    const data = await res.json();
    //console.log(data);
    //alert(data);

    tbody.innerHTML = "";

    data.data.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = "<td>" + r.code + "<td>" +
            "<td>" + r.sang + "<td>" +
            "<td>" + r.su + "<td>" +
            "<td>" + r.dan + "<td>";
        tbody.appendChild(tr);
    });

    clearForm();
    setMsg("조회 완료")
}

// 추가
async function addData(){
    const data = {
        code:Number(code.value),
        code:sang,value,
        code:Number(su.value),
        code:Number(dan,value)
    };



    const res = await fetch("/api/sangdata", {
        method:"POST", 
        headers:{"Content-Type":"application/json"},
        body : JSON.stringify(data)     // js 객체를 문자열로 변환해 전송
    });

    await res.json();
    setMsg("추가 완료");
    clearForm();
    loadAll();  // 추가 후 전체 자료 보기
}





window.onload = loadAll;

btnAdd.onclick = addData;
btnUpdate.onclick = updateData;
btnDelete.onclick = deleteData;
btnReload.onclick = loadAll;