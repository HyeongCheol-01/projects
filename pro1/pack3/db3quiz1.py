# 문1) jikwon 테이블 자료 출력
# 키보드로부터 부서번호를 입력받아, 해당 부서에 직원 자료 출력
# 부서번호 입력 : _______
# 직원번호 직원명 근무지역 직급
# 1 홍길동 서울 이사
# ...
# 인원 수 :


import MySQLdb
import pickle

"""
config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8'
    }
"""

with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)

def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        bu_no = input('부서번호 입력:')
        # print(bu_no)
        sql = """
        select jikwonno 직원번호, jikwonname 직원명, 
        buserloc 근무지역, jikwonjik 직급
        from jikwon
        inner join buser on busernum=buserno
        where busernum={0}
        """.format(bu_no)
        # print(sql)
        
        cursor.execute(sql)

        datas = cursor.fetchall()
        # print(datas)

        if len(datas)==0:
            print(bu_no + "번 부서는 없어요")
            return   # sys.exit(0)

        for jikwonno, jikwonname , buserloc , jikwonjik in datas:
            print(jikwonno, jikwonname , buserloc , jikwonjik)

        print('인원 수 : ' +  str(len(datas)))

    except Exception as e:
        print('err :', e)
    finally:
        cursor.close()
        conn.close()


if __name__=="__main__":
    chulbal()
