"""
01_seed_mariadb.py

MariaDB에 교육기관 상담 챗봇용 샘플 DB와 테이블, 데이터를 생성합니다.

실행:
python scripts/01_seed_mariadb.py
"""

import sys
from pathlib import Path

# scripts/에서 실행해도 app 패키지를 import할 수 있도록 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pymysql
from app.config import settings
from app.db import get_connection


def create_database_if_not_exists() -> None:
    """DB가 없으면 생성합니다."""
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        charset="utf8mb4",
        autocommit=True,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS {settings.db_name} "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()


def create_tables() -> None:
    """수강 과정, 교육생, 출결, 평가, 상담 이력 테이블을 생성합니다."""
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS courses (
            course_id INT PRIMARY KEY AUTO_INCREMENT,
            course_name VARCHAR(100) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            completion_attendance_rate DECIMAL(5,2) NOT NULL,
            completion_avg_score DECIMAL(5,2) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(30) NOT NULL,
            phone_masked VARCHAR(30),
            status VARCHAR(20) NOT NULL,
            course_id INT NOT NULL,
            FOREIGN KEY (course_id) REFERENCES courses(course_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS attendances (
            attendance_id INT PRIMARY KEY AUTO_INCREMENT,
            student_id INT NOT NULL,
            attend_date DATE NOT NULL,
            attendance_rate DECIMAL(5,2) NOT NULL,
            absent_count INT NOT NULL DEFAULT 0,
            late_count INT NOT NULL DEFAULT 0,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_id INT PRIMARY KEY AUTO_INCREMENT,
            student_id INT NOT NULL,
            eval_name VARCHAR(100) NOT NULL,
            score DECIMAL(5,2) NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS counseling_notes (
            note_id INT PRIMARY KEY AUTO_INCREMENT,
            student_id INT NOT NULL,
            note_date DATE NOT NULL,
            counselor VARCHAR(30) NOT NULL,
            note TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """,
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            for ddl in ddl_statements:
                cur.execute(ddl)


def clear_tables() -> None:
    """수업 중 반복 실행할 수 있도록 기존 샘플 데이터를 삭제합니다."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            for table in [
                "counseling_notes",
                "evaluations",
                "attendances",
                "students",
                "courses",
            ]:
                cur.execute(f"TRUNCATE TABLE {table}")
            cur.execute("SET FOREIGN_KEY_CHECKS = 1")


def insert_sample_data() -> None:
    """교육용 샘플 데이터를 입력합니다."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 과정 데이터
            cur.execute(
                """
                INSERT INTO courses
                (course_name, start_date, end_date, completion_attendance_rate, completion_avg_score)
                VALUES (%s, %s, %s, %s, %s)
                """,
                ("AI MLOps 엔지니어 양성과정", "2026-07-01", "2026-12-31", 80.0, 60.0),
            )
            course_id = cur.lastrowid

            # 교육생 데이터
            students = [
                ("홍길동", "010-****-1234", "훈련중", course_id),
                ("김민수", "010-****-5678", "훈련중", course_id),
                ("이영희", "010-****-9012", "훈련중", course_id),
            ]
            cur.executemany(
                """
                INSERT INTO students (name, phone_masked, status, course_id)
                VALUES (%s, %s, %s, %s)
                """,
                students,
            )

            # 교육생 ID 조회
            cur.execute("SELECT student_id, name FROM students")
            student_map = {row["name"]: row["student_id"] for row in cur.fetchall()}

            # 출결 데이터: 주차별 누적 출석률 느낌의 샘플
            attendances = [
                # 홍길동: 수료 기준 80%를 조금 넘는 상태
                (student_map["홍길동"], "2026-07-31", 84.0, 1, 2),
                (student_map["홍길동"], "2026-08-31", 82.0, 2, 3),
                (student_map["홍길동"], "2026-09-30", 80.0, 3, 4),
                # 김민수: 출석률 부족
                (student_map["김민수"], "2026-07-31", 76.0, 4, 2),
                (student_map["김민수"], "2026-08-31", 74.0, 5, 3),
                (student_map["김민수"], "2026-09-30", 72.0, 6, 4),
                # 이영희: 안정적
                (student_map["이영희"], "2026-07-31", 94.0, 0, 1),
                (student_map["이영희"], "2026-08-31", 92.0, 0, 1),
                (student_map["이영희"], "2026-09-30", 90.0, 1, 1),
            ]
            cur.executemany(
                """
                INSERT INTO attendances
                (student_id, attend_date, attendance_rate, absent_count, late_count)
                VALUES (%s, %s, %s, %s, %s)
                """,
                attendances,
            )

            # 평가 데이터
            evaluations = [
                (student_map["홍길동"], "Python 기초 평가", 76.0),
                (student_map["홍길동"], "MLOps 실습 평가", 80.0),
                (student_map["김민수"], "Python 기초 평가", 65.0),
                (student_map["김민수"], "MLOps 실습 평가", 58.0),
                (student_map["이영희"], "Python 기초 평가", 88.0),
                (student_map["이영희"], "MLOps 실습 평가", 91.0),
            ]
            cur.executemany(
                """
                INSERT INTO evaluations (student_id, eval_name, score)
                VALUES (%s, %s, %s)
                """,
                evaluations,
            )

            # 상담 이력
            notes = [
                (student_map["홍길동"], "2026-08-05", "박상담", "프로젝트 주제 선정은 완료했으나 출결 관리가 필요함."),
                (student_map["홍길동"], "2026-09-10", "박상담", "출석률 80% 초반으로 유지 중이므로 결석 최소화 안내."),
                (student_map["김민수"], "2026-08-06", "박상담", "개인 사정으로 결석이 누적되어 보강 계획 수립 필요."),
                (student_map["김민수"], "2026-09-12", "박상담", "평가 점수와 출석률 모두 보완 필요. 멘토링 권장."),
                (student_map["이영희"], "2026-08-07", "박상담", "수업 참여도가 높고 포트폴리오 완성도가 우수함."),
            ]
            cur.executemany(
                """
                INSERT INTO counseling_notes (student_id, note_date, counselor, note)
                VALUES (%s, %s, %s, %s)
                """,
                notes,
            )


def main() -> None:
    create_database_if_not_exists()
    create_tables()
    clear_tables()
    insert_sample_data()
    print("MariaDB 샘플 데이터 생성 완료")


if __name__ == "__main__":
    main()
