from django.urls import path

from authApp.views import LoginView, LogoutView, ProfileView, RefreshTokenView, SignupView, VerifyOTPView, ResetPasswordView, ForgotPasswordView, CurrentUserLoginHistoryView
from authApp import views
urlpatterns = [
    # Mobile Auth Urls
    # path('v1/mobile/auth/signup', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view()),
    path('signup/', SignupView.as_view()),
    path('logout/', LogoutView.as_view()),
    path('forgot-password/',ForgotPasswordView.as_view()),
    path('verify-otp/', VerifyOTPView.as_view()),
    path('reset-password/', ResetPasswordView.as_view()),
    path('profile/', ProfileView.as_view()),
    path('refresh/', RefreshTokenView.as_view()),
    # Web Auth Urls
    path('web/user/', views.__get_current_user, name='get_user'),
    path('web/logout/', views.__logout_current_user, name='logout'),
    path('web/login-history/', CurrentUserLoginHistoryView.as_view(), name='login-history'),
]
