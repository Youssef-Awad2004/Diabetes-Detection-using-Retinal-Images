import { Shield, Zap, FileCheck, Clock } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

const features = [
  {
    icon: Zap,
    title: "Fast Analysis",
    description: "Get classification results in seconds with our AI-powered screening system",
  },
  {
    icon: Shield,
    title: "Secure Processing",
    description: "Your medical images are processed securely and never stored permanently",
  },
  {
    icon: FileCheck,
    title: "Detailed Reports",
    description: "Receive comprehensive analysis with confidence scores and recommendations",
  },
  {
    icon: Clock,
    title: "Early Detection",
    description: "Identify diabetic retinopathy at early stages for better treatment outcomes",
  },
]

export function InfoSection() {
  return (
    <section className="mt-12 border-t pt-12">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground">
          How It Works
        </h2>
        <p className="mx-auto mt-2 max-w-2xl text-muted-foreground">
          Our AI screening tool analyzes retinal fundus images to detect signs of diabetic
          retinopathy, helping healthcare providers make informed decisions.
        </p>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {features.map((feature) => (
          <Card key={feature.title} className="text-center">
            <CardContent className="pt-6">
              <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-primary/10">
                <feature.icon className="size-6 text-primary" />
              </div>
              <h3 className="mt-4 font-semibold text-foreground">{feature.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-12 rounded-xl border bg-muted/30 p-6">
        <h3 className="text-lg font-semibold text-foreground">
          Classification Categories
        </h3>
        <p className="mt-1 text-sm text-muted-foreground">
          The system classifies retinal images into the following diabetic retinopathy stages:
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <div className="rounded-lg bg-success/10 p-3">
            <span className="text-sm font-medium text-success">Normal</span>
            <p className="mt-1 text-xs text-muted-foreground">No DR signs</p>
          </div>
          <div className="rounded-lg bg-chart-2/10 p-3">
            <span className="text-sm font-medium text-chart-2">Mild NPDR</span>
            <p className="mt-1 text-xs text-muted-foreground">Early changes</p>
          </div>
          <div className="rounded-lg bg-severe/10 p-3">
            <span className="text-sm font-medium text-severe">Severe NPDR</span>
            <p className="mt-1 text-xs text-muted-foreground">Advanced</p>
          </div>
        </div>
      </div>
    </section>
  )
}
