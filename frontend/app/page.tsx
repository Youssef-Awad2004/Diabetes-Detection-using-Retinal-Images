import { Header } from "@/components/retinal-analysis/header"
import { AnalysisForm } from "@/components/retinal-analysis/analysis-form"
import { InfoSection } from "@/components/retinal-analysis/info-section"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Hero Section */}
        <div className="mb-10 text-center">
          <h1 className="text-balance text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Diabetic Retinopathy Screening
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-pretty text-lg text-muted-foreground">
            Upload a retinal fundus image for AI-powered classification. Our system analyzes
            the image and provides a diabetic retinopathy severity assessment.
          </p>
        </div>

        {/* Analysis Form */}
        <AnalysisForm />

        {/* Info Section */}
        <InfoSection />
      </main>

      {/* Footer */}
      <footer className="border-t bg-card mt-12">
        <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
            <p className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} RetinaScreen. For research and screening purposes only.
            </p>
            <div className="flex items-center gap-6">
              <a
                href="#"
                className="text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                Privacy Policy
              </a>
              <a
                href="#"
                className="text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                Terms of Service
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
