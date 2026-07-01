from django.urls import path
from .views import (
    AskQuestionAPIView,
    QuestionHistoryListAPIView,
    delete_history,
    list_sessions,
    delete_session,
    evaluate,
    compare_methods,
)


urlpatterns = [
    path("ask/", AskQuestionAPIView.as_view(), name="ask-question"),
    path("history/", QuestionHistoryListAPIView.as_view(), name="question-history"),
    path("history/<int:history_id>/delete/", delete_history, name="delete-history"),
    path("sessions/", list_sessions, name="list-sessions"),
    path("sessions/<int:session_id>/delete/", delete_session, name="delete-session"),
    path("evaluate/", evaluate, name="evaluate"),
    path("compare/", compare_methods, name="compare-methods"),
]
