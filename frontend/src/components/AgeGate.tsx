import { useCallback, useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import './AgeGate.css'

const COOKIE_NAME = 'vino_age_confirmed'
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365 // 1 year

/** Legal/info pages linked from the gate — readable without confirming 18+. */
export const AGE_GATE_EXEMPT_PATHS = new Set(['/cookies', '/recommendations'])

function readConfirmed(): boolean {
  if (typeof document === 'undefined') return false
  return document.cookie.split(';').some((part) => {
    const [k, v] = part.trim().split('=')
    return k === COOKIE_NAME && (v === '1' || v === 'true')
  })
}

function writeConfirmed() {
  document.cookie = `${COOKIE_NAME}=1; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`
}

export function AgeGate() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  const exempt = AGE_GATE_EXEMPT_PATHS.has(pathname)
  const show = open && !exempt

  useEffect(() => {
    setOpen(!readConfirmed())
  }, [])

  useEffect(() => {
    if (!show) {
      document.body.style.overflow = ''
      return
    }
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [show])

  const confirm = useCallback(() => {
    writeConfirmed()
    setOpen(false)
  }, [])

  if (!show) return null

  return (
    <div className="age-gate" role="dialog" aria-modal="true" aria-labelledby="age-gate-title">
      <div className="age-gate__card">
        <h2 id="age-gate-title" className="age-gate__title">
          18+
        </h2>
        <p className="age-gate__text">
          Для просмотра сайта необходимо подтвердить своё совершеннолетие и согласие
          на обработку файлов{' '}
          <Link className="age-gate__link" to="/cookies">
            cookies
          </Link>{' '}
          и использование{' '}
          <Link className="age-gate__link" to="/recommendations">
            рекомендательных технологий
          </Link>
        </p>
        <button type="button" className="age-gate__btn" onClick={confirm}>
          Подтверждаю
        </button>
      </div>
    </div>
  )
}
