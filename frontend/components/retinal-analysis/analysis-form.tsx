"use client"

import * as React from "react"
import { Loader2, Scan, RotateCcw } from "lucide-react"
import { ImageUploader } from "./image-uploader"
import { ClassificationResultDisplay } from "./classification-result"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { API_CONFIG, type ClassificationResult, type APIResponse, mapBackendResponse } from "@/lib/api-config"

type AnalysisState = "idle" | "uploading" | "analyzing" | "complete" | "error"

async function classifyImage(file: File): Promise<APIResponse> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(API_CONFIG.CLASSIFICATION_API_URL, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Server error: ${response.status} ${response.statusText}`)
  }

  const data = await response.json()
  return mapBackendResponse(data)
}

export function AnalysisForm() {
  const [selectedImage, setSelectedImage] = React.useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [analysisState, setAnalysisState] = React.useState<AnalysisState>("idle")
  const [progress, setProgress] = React.useState(0)
  const [result, setResult] = React.useState<ClassificationResult | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    setResult(null)
    setError(null)
    setAnalysisState("idle")

    // Create preview URL
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleImageClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setSelectedImage(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setAnalysisState("idle")
    setProgress(0)
  }

  const handleAnalyze = async () => {
    if (!selectedImage) return

    setAnalysisState("uploading")
    setProgress(0)
    setError(null)

    try {
      // Simulate upload progress
      const uploadInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 40) {
            clearInterval(uploadInterval)
            return 40
          }
          return prev + 8
        })
      }, 100)

      // Wait for upload simulation
      await new Promise((resolve) => setTimeout(resolve, 600))
      clearInterval(uploadInterval)
      setProgress(40)

      setAnalysisState("analyzing")

      // Simulate analysis progress
      const analyzeInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(analyzeInterval)
            return 90
          }
          return prev + 5
        })
      }, 150)

      // Call the classification API (mock for now)
      const response = await classifyImage(selectedImage)

      clearInterval(analyzeInterval)
      setProgress(100)

      if (response.success && response.result) {
        setResult(response.result)
        setAnalysisState("complete")
      } else {
        throw new Error(response.error || "Classification failed")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred")
      setAnalysisState("error")
      setProgress(0)
    }
  }

  const handleReset = () => {
    handleImageClear()
  }

  const isProcessing = analysisState === "uploading" || analysisState === "analyzing"

  return (
    <div className="grid gap-8 lg:grid-cols-2">
      {/* Left Column - Upload Section */}
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Step 1: Upload Image</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Select a retinal fundus image for diabetic retinopathy screening
          </p>
        </div>

        <ImageUploader
          onImageSelect={handleImageSelect}
          onImageClear={handleImageClear}
          selectedImage={selectedImage}
          previewUrl={previewUrl}
          disabled={isProcessing}
        />

        {/* Progress Indicator */}
        {isProcessing && (
          <div className="space-y-3 rounded-lg border bg-card p-4">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-foreground">
                {analysisState === "uploading" ? "Uploading image..." : "Analyzing image..."}
              </span>
              <span className="text-muted-foreground">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
            <p className="text-xs text-muted-foreground">
              {analysisState === "uploading"
                ? "Preparing image for analysis"
                : "AI model processing retinal patterns"}
            </p>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3">
          <Button
            onClick={handleAnalyze}
            disabled={!selectedImage || isProcessing}
            size="lg"
            className="flex-1 sm:flex-none"
          >
            {isProcessing ? (
              <>
                <Loader2 className="animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Scan />
                Analyze Image
              </>
            )}
          </Button>

          {(result || error) && (
            <Button
              onClick={handleReset}
              variant="outline"
              size="lg"
              disabled={isProcessing}
            >
              <RotateCcw />
              New Analysis
            </Button>
          )}
        </div>
      </div>

      {/* Right Column - Results Section */}
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Step 2: View Results</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Classification results and recommendations will appear here
          </p>
        </div>

        <ClassificationResultDisplay result={result} isLoading={isProcessing} />
      </div>
    </div>
  )
}
