/** Per-browser preference: show detailed search UI. Default off. */

export const SHOW_SEARCH_DETAILS_COOKIE = 'vino_show_search_details'
export const SHOW_SEARCH_DETAILS_EVENT = 'vino:show-search-details'

export function readShowSearchDetailsCookie(): boolean {
  if (typeof document === 'undefined') return false
  const match = document.cookie.match(
    /(?:^|;\s*)vino_show_search_details=([^;]*)/,
  )
  const raw = match ? decodeURIComponent(match[1]) : null
  return raw === '1' || raw === 'true'
}

export function writeShowSearchDetailsCookie(on: boolean) {
  const maxAge = 60 * 60 * 24 * 365
  document.cookie = `${SHOW_SEARCH_DETAILS_COOKIE}=${on ? '1' : '0'}; path=/; max-age=${maxAge}; SameSite=Lax`
}

export function setShowSearchDetailsPreference(on: boolean) {
  writeShowSearchDetailsCookie(on)
  window.dispatchEvent(
    new CustomEvent(SHOW_SEARCH_DETAILS_EVENT, { detail: { show: on } }),
  )
}
