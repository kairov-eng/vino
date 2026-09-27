/** Re-export site password gate + role context. */
export {
  SITE_PASSWORD_COOKIE,
  SiteAuthProvider as SiteAccessGate,
  useSiteAuth,
  type SiteRole,
} from '../auth/SiteAuthContext'
