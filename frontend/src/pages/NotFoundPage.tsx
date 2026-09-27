import { Link, useLocation } from 'react-router-dom'
import './NotFoundPage.css'

export function NotFoundPage() {
  const { pathname } = useLocation()
  const path = pathname || '/'

  return (
    <main className="error-page">
      <div className="error-page__content">
        <h1 className="error-page__code">404</h1>
        <p className="error-page__message">Page not found: {path}</p>
        <Link to="/" className="error-page__btn">
          Вернуться на главную
        </Link>
      </div>
    </main>
  )
}
