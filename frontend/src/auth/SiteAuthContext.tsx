import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type FormEvent,
  type ReactNode,
} from 'react'
import { checkSiteAuth, loginSiteAuth } from '../api/client'
import '../components/SiteAccessGate.css'

export const SITE_PASSWORD_COOKIE = 'vino_site_password'
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365 // 1 year

export type SiteRole = 'admin' | 'user' | null

type SiteAuthContextValue = {
  role: SiteRole
  /** Site password gate is active (PASSWORD_* configured). */
  enabled: boolean
  /** Admin UI (gear, search details). True when gate is disabled. */
  isAdmin: boolean
}

const SiteAuthContext = createContext<SiteAuthContextValue>({
  role: null,
  enabled: false,
  isAdmin: true,
})

export function useSiteAuth(): SiteAuthContextValue {
  return useContext(SiteAuthContext)
}

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

export function SiteAuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false)
  const [unlocked, setUnlocked] = useState(false)
  const [role, setRole] = useState<SiteRole>(null)
  const [enabled, setEnabled] = useState(false)
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
        setEnabled(Boolean(res.enabled))
        if (!res.enabled) {
          setRole(null)
          setUnlocked(true)
          setReady(true)
          return
        }
        if (res.ok) {
          setRole(res.role === 'admin' || res.role === 'user' ? res.role : null)
          setUnlocked(true)
        } else if (stored) {
          clearPasswordCookie()
        }
      } catch {
        if (!cancelled) {
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
        setEnabled(Boolean(res.enabled))
        setRole(res.role === 'admin' || res.role === 'user' ? res.role : null)
        setUnlocked(true)
      } catch {
        setError('Неверный пароль')
      } finally {
        setBusy(false)
      }
    },
    [password],
  )

  const value = useMemo<SiteAuthContextValue>(() => {
    const isAdmin = !enabled || role === 'admin'
    return { role, enabled, isAdmin }
  }, [role, enabled])

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

  return (
    <SiteAuthContext.Provider value={value}>{children}</SiteAuthContext.Provider>
  )
}
