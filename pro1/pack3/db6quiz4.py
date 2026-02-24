# 문3) 성별 직원 현황 출력 : 성별(남/여) 단위로 직원 수와 평균 급여 출력
# 성별 직원수 평균급여
# 남 3 8500
# 여 2 7800
import MySQLdb
import pickle

with open('mydb.dat', mode='rb') as obj:
    config = pickle.load(obj)

def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()


        sql = """
        select jikwonno 직원번호, jikwonname 직원명, count(gogekno) 관리고객수
        from jikwon
        right outer join gogek on jikwonno = gogekdamsano
        group by jikwonno
        """
        cursor.execute(sql)
        datas = cursor.fetchall()

        for a,b,c in datas:
            print(a,b,c)


    except Exception as e:
        print('err :', e)
    finally:
        cursor.close()
        conn.close()


if __name__=="__main__":
    chulbal()
