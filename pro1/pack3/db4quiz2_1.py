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
        where busernum=(select busernum from jikwon where jikwonno={0} and jikwonname='{1}')
        order by jikwonjik,jikwonname
        """.format(jik_no, jik_name)
        cursor.execute(sql)
        datas = cursor.fetchall()

        if len(datas)==0:
            print('해당직원 없음')
            return
        
        for jikwonno,jikwonname,busername,busertel,jikwonjik,jikwongen in datas:
            print(jikwonno,jikwonname,busername,busertel,jikwonjik,jikwongen)
        print('직원 수 : ' +  str(len(datas)))

        print()
        
        sql2 = """
        select gogekno 고객번호, gogekname 고객명, gogektel 고객전화, Date_format(now(), "%y")-left(gogekjumin,2)+100 나이
        from jikwon
        inner join gogek on jikwonno=gogekdamsano
        where jikwonno={0}
        """.format(jik_no)
        cursor.execute(sql2)
        datas2 = cursor.fetchall()
        if len(datas2)==0:
            print('담당고객 없음')
            return

        for gogekno, gogekname, gogektel, nai in datas2:
            print(gogekno, gogekname, gogektel, nai)
        print('관리 고객 수 : ' +  str(len(datas2)))


    except Exception as e:
        print('err :', e)
    finally:
        cursor.close()
        conn.close()


if __name__=="__main__":
    chulbal()
