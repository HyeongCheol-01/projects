"""
MariaDB 접근 모듈

현업 팁:
- LLM이 직접 SQL을 마음대로 만들게 하면 위험할 수 있습니다.
- 이 예제는 교육용/실무형 안전 구조를 보여주기 위해
  미리 정한 SELECT 쿼리만 실행합니다.
"""

from typing import Any
import pymysql
from pymysql.cursors import DictCursor
from langchain_community.utilities import SQLDatabase

from app.config import settings


def get_connection(database: str | None = None):
    """MariaDB 연결 객체를 생성합니다."""
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=database or settings.db_name,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )


def get_langchain_sql_database() -> SQLDatabase:
    """
    LangChain의 SQLDatabase 객체를 생성합니다.

    이 객체는 DB 스키마 확인, SQL Agent 구성 등에 사용할 수 있습니다.
    이번 예제에서는 안전을 위해 실제 조회는 고정 SQL로 처리합니다.
    """
    return SQLDatabase.from_uri(settings.sqlalchemy_uri)


def fetch_all_student_names() -> list[str]:
    """질문에서 교육생 이름을 찾기 위해 전체 교육생 이름을 가져옵니다."""
    sql = "SELECT name FROM students ORDER BY name"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return [row["name"] for row in rows]


def fetch_student_report(student_name: str) -> dict[str, Any] | None:
    """
    특정 교육생의 출결, 평가, 상담 정보를 요약해서 가져옵니다.
    """
    sql = """
        SELECT
            s.student_id,
            s.name AS student_name,
            s.status,
            c.course_name,
            c.start_date,
            c.end_date,
            c.completion_attendance_rate,
            c.completion_avg_score,
            ROUND(AVG(a.attendance_rate), 1) AS attendance_rate,
            SUM(a.absent_count) AS total_absent_count,
            SUM(a.late_count) AS total_late_count,
            ROUND(AVG(e.score), 1) AS avg_score,
            GROUP_CONCAT(
                DISTINCT CONCAT(
                    DATE_FORMAT(n.note_date, '%%Y-%%m-%%d'),
                    ' / ', n.counselor,
                    ' / ', n.note
                )
                ORDER BY n.note_date
                SEPARATOR '\n'
            ) AS counseling_notes
        FROM students s
        JOIN courses c ON s.course_id = c.course_id
        LEFT JOIN attendances a ON s.student_id = a.student_id
        LEFT JOIN evaluations e ON s.student_id = e.student_id
        LEFT JOIN counseling_notes n ON s.student_id = n.student_id
        WHERE s.name = %s
        GROUP BY
            s.student_id,
            s.name,
            s.status,
            c.course_name,
            c.start_date,
            c.end_date,
            c.completion_attendance_rate,
            c.completion_avg_score
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (student_name,))
            row = cur.fetchone()

    return row


def format_student_report(report: dict[str, Any] | None) -> str:
    """LLM 프롬프트에 넣기 좋은 문자열로 DB 조회 결과를 변환합니다."""
    if not report:
        return "해당 교육생 정보를 MariaDB에서 찾지 못했습니다."

    return f"""
[MariaDB 교육생 데이터]
- 교육생명: {report['student_name']}
- 현재 상태: {report['status']}
- 과정명: {report['course_name']}
- 과정 기간: {report['start_date']} ~ {report['end_date']}
- 현재 출석률: {report['attendance_rate']}%
- 총 결석 횟수: {report['total_absent_count']}회
- 총 지각 횟수: {report['total_late_count']}회
- 평균 평가 점수: {report['avg_score']}점
- 과정 수료 기준 출석률: {report['completion_attendance_rate']}%
- 과정 수료 기준 평균 점수: {report['completion_avg_score']}점
- 상담 이력:
{report['counseling_notes'] or '상담 이력 없음'}
""".strip()
