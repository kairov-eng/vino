type InfoStubPageProps = {
  title: string
}

export function InfoStubPage({ title }: InfoStubPageProps) {
  return (
    <main className="page info-stub-page">
      <div className="page__intro">
        <h1>{title}</h1>
        <p className="page__lead">
          Это тестовое задание в рамках хакатона «Лидеры Цифровой Трансформации
          2026».
        </p>
      </div>
    </main>
  )
}
