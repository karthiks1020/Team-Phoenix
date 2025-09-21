document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) || '';
    const apiFetch = (path, options = {}) => {
        const opts = Object.assign({ credentials: 'include' }, options);
        return fetch(`${API_BASE}${path}`, opts);
    };

    const signupFormContainer = document.getElementById('signup-form-container');
    const loginFormContainer = document.getElementById('login-form-container');
    const showLoginLink = document.getElementById('show-login');
    const showSignupLink = document.getElementById('show-signup');
    const showPasswordToggles = document.querySelectorAll('.show-password-toggle');
    const signupPasswordInput = document.getElementById('signup-password');
    const lengthRequirement = document.getElementById('length');
    const caseRequirement = document.getElementById('case');
    const numberRequirement = document.getElementById('number');
    const symbolRequirement = document.getElementById('symbol');
    const signupForm = document.getElementById('signup-form');
    const loginForm = document.getElementById('login-form');
    const forgotPasswordForm = document.getElementById('forgot-password-form');
    const resetPasswordForm = document.getElementById('reset-password-form');
    const profileInfo = document.getElementById('profile-info');
    const updatePhoneForm = document.getElementById('update-phone-form');
    const verifyOtpForm = document.getElementById('verify-otp-form');
    const logoutButton = document.getElementById('logout-button');

    if (showLoginLink) {
        showLoginLink.addEventListener('click', (e) => {
            e.preventDefault();
            signupFormContainer.classList.add('hidden');
            loginFormContainer.classList.remove('hidden');
        });
    }

    if (showSignupLink) {
        showSignupLink.addEventListener('click', (e) => {
            e.preventDefault();
            loginFormContainer.classList.add('hidden');
            signupFormContainer.classList.remove('hidden');
        });
    }

    if (showPasswordToggles) {
        showPasswordToggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                const targetId = toggle.dataset.target;
                const passwordInput = document.getElementById(targetId);

                if (passwordInput.type === 'password') {
                    passwordInput.type = 'text';
                    toggle.textContent = 'Hide';
                } else {
                    passwordInput.type = 'password';
                    toggle.textContent = 'Show';
                }
            });
        });
    }

    if (signupPasswordInput) {
        signupPasswordInput.addEventListener('input', () => {
            const password = signupPasswordInput.value;

            lengthRequirement.classList.toggle('valid', password.length >= 12);
            caseRequirement.classList.toggle('valid', /[a-z]/.test(password) && /[A-Z]/.test(password));
            numberRequirement.classList.toggle('valid', /\d/.test(password));
            symbolRequirement.classList.toggle('valid', /[^a-zA-Z0-9]/.test(password));
        });
    }

    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;

            try {
const res = await apiFetch('/signup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();
                alert(data.message || data.error);
                if (res.ok) window.location.reload();
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;

            try {
const res = await apiFetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();
                alert(data.message || data.error);
                if (res.ok) window.location.href = 'profile.html';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('forgot-email').value;

            try {
const res = await apiFetch('/forgot-password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email })
                });
                const data = await res.json();
                alert(data.message);
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    if (resetPasswordForm) {
        resetPasswordForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const newPassword = document.getElementById('new-password').value;
            const token = new URLSearchParams(window.location.search).get('token');

            if (!token) return alert('No token provided.');

            try {
const res = await apiFetch(`/reset-password/${token}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password: newPassword })
                });
                const data = await res.json();
                alert(data.message || data.error);
                if (res.ok) window.location.href = 'index.html';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    if (profileInfo) {
        const profileEmail = document.getElementById('profile-email');
        const profilePhone = document.getElementById('profile-phone');

        (async () => {
            try {
const res = await apiFetch('/profile');
                if (!res.ok) {
                    window.location.href = 'index.html';
                    return;
                }
                const data = await res.json();
                profileEmail.textContent = data.email;
                profilePhone.textContent = data.phone || 'Not set';
            } catch (error) {
                console.error('Error:', error);
                window.location.href = 'index.html';
            }
        })();
    }

    if (updatePhoneForm) {
        updatePhoneForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const newPhoneNumber = document.getElementById('new-phone-number').value;

            try {
const res = await apiFetch('/profile/update-phone', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ newPhoneNumber })
                });
                const data = await res.json();
                alert(data.message || data.error);
                if (res.ok) {
                    updatePhoneForm.classList.add('hidden');
                    verifyOtpForm.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    if (verifyOtpForm) {
        verifyOtpForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const otp = document.getElementById('otp').value;

            try {
const res = await apiFetch('/profile/verify-phone', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ otp })
                });
                const data = await res.json();
                alert(data.message || data.error);
                if (res.ok) window.location.reload();
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    if (logoutButton) {
        logoutButton.addEventListener('click', async () => {
            try {
const res = await apiFetch('/logout', { method: 'POST' });
                const data = await res.json();
                alert(data.message || data.error);
                if (res.ok) window.location.href = 'index.html';
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }
});