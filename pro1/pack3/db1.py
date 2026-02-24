# 개인용 Database : sqlite3 - 파이썬에 기본 모듈로 제공
# https://www.sqlite.org
# 모바일 기기, 임베디드 시스템 주로 사용.

import sqlite3
print(sqlite3.sqlite_version)

# conn = sqlite3.connect('exam.db')
conn = sqlite3.connect(':memory:')  #RAM에만 db 저장, 휘발성

try:
    cur = conn.cursor();  # sql문 처리를 위한 cursor 객체 생성

    # 테이블 생성
    cur.execute("create table if not exists friends(name text, phone text, addr text)")

    # 자료 입력
    cur.execute("insert into friends values('홍길동','222-2222','서초1동')")
    cur.execute("insert into friends values(?,?,?)", ('신기해','333-3333','역삼2동'))
    inputdatas = ('신기한', '333-4444', '역삼2동')
    cur.execute("insert into friends values(?,?,?)", inputdatas)
    conn.commit()

    #자료 보기
    cur.execute("select * from friends")
    # print(cur.fetchall())  # 한 개의 레코드(행) 읽기
    print()
    cur.execute("select name,addr,phone from friends")
    for r in cur:
        # print(r)
        print(r[0], r[1],r[2])

except Exception as e:
    print('err : ', e)
    conn.rollback()
finally:
    conn.close()
