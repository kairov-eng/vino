import { useCallback, useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { checkSiteAuth, loginSiteAuth } from '../api/client'
import './SiteAccessGate.css'

export const SITE_PASSWORD_COOKIE = 'vino_site_password'
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365 // 1 year

function readPasswordCookie(): string {
  if (typeof document === 'undefined') return ''
  const parts = document.cookie.split(';')
  for (const part of parts) {
    const idx = part.indexOf('=')
    if (idx < 0) continue
    const k = part.slice(0, idx).trim()
    if (k !== SITE_PASSWORD_COOKIE) continue
    try {
      return decodeURIComponent(part.slice(idx + 1).trim())
    } catch {
      return part.slice(idx + 1).trim()
    }
  }
  return ''
}

function writePasswordCookie(password: string) {
  const value = encodeURIComponent(password)
  document.cookie = `${SITE_PASSWORD_COOKIE}=${value}; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`
}

function clearPasswordCookie() {
  document.cookie = `${SITE_PASSWORD_COOKIE}=; path=/; max-age=0; SameSite=Lax`
}

export function SiteAccessGate({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false)
  const [unlocked, setUnlocked] = useState(false)
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const stored = readPasswordCookie()
        const res = await checkSiteAuth(stored || undefined)
        if (cancelled) return
        if (!res.enabled) {
          setUnlocked(true)
          setReady(true)
          return
        }
        if (res.ok) {
          setUnlocked(true)
        } else if (stored) {
          clearPasswordCookie()
        }
      } catch {
        if (!cancelled) {
          // Backend unreachable: still show gate if we have no cookie path
          setUnlocked(false)
        }
      } finally {
        if (!cancelled) setReady(true)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!ready || unlocked) {
      document.body.style.overflow = ''
      return
    }
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [ready, unlocked])

  const onSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()
      const value = password.trim()
      if (!value) {
        setError('Введите пароль')
        return
      }
      setBusy(true)
      setError(null)
      try {
        const res = await loginSiteAuth(value)
        if (!res.ok) {
          setError('Неверный пароль')
          return
        }
        writePasswordCookie(value)
        setUnlocked(true)
      } catch {
        setError('Неверный пароль')
      } finally {
        setBusy(false)
      }
    },
    [password],
  )

  if (!ready) {
    return (
      <div className="site-access-gate site-access-gate--loading" aria-busy="true">
        <p>Загрузка…</p>
      </div>
    )
  }

  if (!unlocked) {
    return (
      <div
        className="site-access-gate"
        role="dialog"
        aria-modal="true"
        aria-labelledby="site-access-title"
      >
        <form className="site-access-gate__card" onSubmit={onSubmit}>
          <h2 id="site-access-title" className="site-access-gate__title">
            Доступ
          </h2>
          <p className="site-access-gate__text">
            Сайт закрыт. Введите пароль, чтобы продолжить.
          </p>
          <label className="site-access-gate__label">
            <span className="site-access-gate__label-text">Пароль</span>
            <input
              className="site-access-gate__input"
              type="password"
              name="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={busy}
              autoFocus
            />
          </label>
          {error && <p className="site-access-gate__error">{error}</p>}
          <button
            type="submit"
            className="site-access-gate__btn"
            disabled={busy}
          >
            {busy ? 'Проверка…' : 'Войти'}
          </button>
        </form>
      </div>
    )
  }

  return <>{children}</>
}
