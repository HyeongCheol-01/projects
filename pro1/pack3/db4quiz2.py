# 문2) 직원번호와 직원명을 입력(로그인)하여 성공하면 아래의 내용 출력
# 직원번호 입력 : _______
# 직원명 입력 : _______
# 직원번호 직원명 부서명 부서전화 직급 성별
# 1 홍길동 총무부 111-1111 이사 남
import MySQLdb
import pickle

with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)

def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        
        jik_no = input('직원번호 입력 : ')
        jik_name = input('직원명 입력 : ')

        sql = """
        select jikwonno 직원번호, jikwonname 직원명, 
        busername 부서명, busertel 부서전화,
        jikwonjik 직급, jikwongen 성별
        from jikwon
        inner join buser on busernum=buserno
        where jikwonno={0} and jikwonname='{1}'
        """.format(jik_no, jik_name)
        
        cursor.execute(sql)
        datas = cursor.fetchone()

        print(datas[0],datas[1],datas[2],datas[3],datas[4],datas[5])

    except Exception as e:
        print('err :', e)
    finally:
        cursor.close()
        conn.close()


if __name__=="__main__":
    chulbal()
