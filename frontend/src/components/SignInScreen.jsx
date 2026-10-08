function SignInScreen({ error }) {
  return (
    <div className="auth-page">
      <section className="auth-card">
        <a className="brand" href="/" aria-label="ARIA home">
          <span className="brand-mark">A</span><span>ARIA</span>
        </a>
        <p className="eyebrow">Private interview practice</p>
        <h1>Your interviews, securely connected.</h1>
        <p>Sign in to keep each résumé, job description, question, and answer attached to your account.</p>
        {error && <p className="form-error" role="alert">{error}</p>}
        <a className="primary-button auth-button" href="/api/auth/google/start">
          <span>Continue with Google</span><span aria-hidden="true">→</span>
        </a>
      </section>
    </div>
  );
}

export default SignInScreen;
