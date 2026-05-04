from django.urls import path
from .views import AnalyzePostView, HistoryView, AnalysisDetailView, HealthCheckView

urlpatterns = [
    path('analyze', AnalyzePostView.as_view(), name='analyze'),
    path('history', HistoryView.as_view(), name='history'),
    path('history/<str:analysis_id>', AnalysisDetailView.as_view(), name='analysis_detail'),
    path('health', HealthCheckView.as_view(), name='health_check'),
]
