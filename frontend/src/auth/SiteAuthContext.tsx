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
import { checkAdminAuth, verifyAdminPassword } from '../api/client'
import '../components/SiteAccessGate.css'

/** Cookie with admin password for PUT /api/settings. */
export const ADMIN_PASSWORD_COOKIE = 'vino_admin_password'
const LEGACY_SITE_PASSWORD_COOKIE = 'vino_site_password'
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365 // 1 year

type SiteAuthContextValue = {
  /** Admin password is configured on the server (edits need unlock). */
  passwordRequired: boolean
  /** Settings form is unlocked for editing in this session. */
  canEdit: boolean
  /** Unlock edits: uses cookie if valid, otherwise caller shows password UI. */
  tryUnlockFromCookie: () => Promise<boolean>
  unlockWithPassword: (password: string) => Promise<boolean>
  lockEdit: () => void
  readAdminPassword: () => string
}

const SiteAuthContext = createContext<SiteAuthContextValue>({
  passwordRequired: false,
  canEdit: true,
  tryUnlockFromCookie: async () => true,
  unlockWithPassword: async () => true,
  lockEdit: () => undefined,
  readAdminPassword: () => '',
})

export function useSiteAuth(): SiteAuthContextValue {
  return useContext(SiteAuthContext)
}

function readCookie(name: string): string {
  if (typeof document === 'undefined') return ''
  const parts = document.cookie.split(';')
  for (const part of parts) {
    const idx = part.indexOf('=')
    if (idx < 0) continue
    const k = part.slice(0, idx).trim()
    if (k !== name) continue
    try {
      return decodeURIComponent(part.slice(idx + 1).trim())
    } catch {
      return part.slice(idx + 1).trim()
    }
  }
  return ''
}

function writeAdminPasswordCookie(password: string) {
  const value = encodeURIComponent(password)
  document.cookie = `${ADMIN_PASSWORD_COOKIE}=${value}; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`
}

function clearAdminPasswordCookie() {
  document.cookie = `${ADMIN_PASSWORD_COOKIE}=; path=/; max-age=0; SameSite=Lax`
  document.cookie = `${LEGACY_SITE_PASSWORD_COOKIE}=; path=/; max-age=0; SameSite=Lax`
}

export function readAdminPasswordCookie(): string {
  return (
    readCookie(ADMIN_PASSWORD_COOKIE) || readCookie(LEGACY_SITE_PASSWORD_COOKIE)
  )
}

export function SiteAuthProvider({ children }: { children: ReactNode }) {
  const [passwordRequired, setPasswordRequired] = useState(false)
  const [canEdit, setCanEdit] = useState(false)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await checkAdminAuth()
        if (cancelled) return
        const required = Boolean(res.enabled)
        setPasswordRequired(required)
        if (!required) {
          setCanEdit(true)
        }
      } catch {
        if (!cancelled) {
          // Fail open for read-only browsing; edits still need server check.
          setPasswordRequired(true)
          setCanEdit(false)
        }
      } finally {
        if (!cancelled) setReady(true)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const unlockWithPassword = useCallback(async (password: string) => {
    const value = password.trim()
    if (!value) return false
    const res = await verifyAdminPassword(value)
    if (!res.ok) {
      clearAdminPasswordCookie()
      setCanEdit(false)
      return false
    }
    writeAdminPasswordCookie(value)
    setCanEdit(true)
    return true
  }, [])

  const tryUnlockFromCookie = useCallback(async () => {
    if (!passwordRequired) {
      setCanEdit(true)
      return true
    }
    const stored = readAdminPasswordCookie()
    if (!stored) return false
    const res = await verifyAdminPassword(stored)
    if (!res.ok) {
      clearAdminPasswordCookie()
      setCanEdit(false)
      return false
    }
    // Refresh cookie under the new name if legacy was used.
    writeAdminPasswordCookie(stored)
    setCanEdit(true)
    return true
  }, [passwordRequired])

  const lockEdit = useCallback(() => {
    setCanEdit(false)
  }, [])

  const value = useMemo<SiteAuthContextValue>(
    () => ({
      passwordRequired,
      canEdit: passwordRequired ? canEdit : true,
      tryUnlockFromCookie,
      unlockWithPassword,
      lockEdit,
      readAdminPassword: readAdminPasswordCookie,
    }),
    [
      passwordRequired,
      canEdit,
      tryUnlockFromCookie,
      unlockWithPassword,
      lockEdit,
    ],
  )

  if (!ready) {
    return (
      <div className="site-access-gate site-access-gate--loading" aria-busy="true">
        <p>Загрузка…</p>
      </div>
    )
  }

  return (
    <SiteAuthContext.Provider value={value}>{children}</SiteAuthContext.Provider>
  )
}

/** Small password form used inside settings unlock flow. */
export function AdminPasswordForm({
  busy,
  error,
  onSubmit,
  onCancel,
}: {
  busy?: boolean
  error?: string | null
  onSubmit: (password: string) => void | Promise<void>
  onCancel?: () => void
}) {
  const [password, setPassword] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault()
      const value = password.trim()
      if (!value) {
        setLocalError('Введите пароль')
        return
      }
      setLocalError(null)
      await onSubmit(value)
    },
    [password, onSubmit],
  )

  return (
    <form className="site-access-gate__card site-access-gate__card--inline" onSubmit={handleSubmit}>
      <h2 className="site-access-gate__title" style={{ fontSize: '1.5rem' }}>
        Пароль администратора
      </h2>
      <p className="site-access-gate__text">
        Чтобы изменить настройки поиска, введите пароль администратора.
      </p>
      <label className="site-access-gate__label">
        <span className="site-access-gate__label-text">Пароль</span>
        <input
          className="site-access-gate__input"
          type="password"
          name="admin_password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={busy}
          autoFocus
        />
      </label>
      {(localError || error) && (
        <p className="site-access-gate__error">{localError || error}</p>
      )}
      <div className="site-access-gate__actions">
        {onCancel && (
          <button
            type="button"
            className="site-access-gate__btn site-access-gate__btn--ghost"
            onClick={onCancel}
            disabled={busy}
          >
            Отмена
          </button>
        )}
        <button type="submit" className="site-access-gate__btn" disabled={busy}>
          {busy ? 'Проверка…' : 'Разблокировать'}
        </button>
      </div>
    </form>
  )
}
