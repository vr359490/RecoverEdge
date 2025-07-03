//
//  GuidedRecoveryView.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 7/1/25.
//

import SwiftUI

// MARK: - Guided Recovery Session View
struct GuidedRecoveryView: View {
    let methods: [RecoveryMethod]
    let totalTime: Int
    @Environment(\.presentationMode) var presentationMode
    @State private var currentStepIndex = 0
    @State private var timeRemaining = 0
    @State private var isTimerRunning = false
    @State private var sessionComplete = false
    @State private var showingPauseMenu = false
    @State private var timer: Timer?
    
    var currentMethod: RecoveryMethod {
        methods[currentStepIndex]
    }
    
    var isLastStep: Bool {
        currentStepIndex == methods.count - 1
    }
    
    var body: some View {
        NavigationView {
            ZStack {
                // Main content
                if sessionComplete {
                    SessionCompleteView(
                        totalMethods: methods.count,
                        totalTime: totalTime,
                        onDismiss: { presentationMode.wrappedValue.dismiss() }
                    )
                } else {
                    VStack(spacing: 0) {
                        // Progress Header
                        ProgressHeaderView(
                            currentStep: currentStepIndex + 1,
                            totalSteps: methods.count,
                            onPause: { showingPauseMenu = true }
                        )
                        
                        // Main Step Content
                        ScrollView {
                            VStack(spacing: 24) {
                                // Timer Circle
                                TimerCircleView(
                                    timeRemaining: timeRemaining,
                                    totalTime: currentMethod.duration * 60,
                                    isRunning: isTimerRunning
                                )
                                .padding(.top, 20)
                                
                                // Method Details
                                MethodDetailsView(method: currentMethod)
                                
                                // Video Player (if available)
                                if let vimeoID = currentMethod.videoURL, !vimeoID.isEmpty {
                                    VStack(alignment: .leading, spacing: 12) {
                                        Text("Watch Demonstration")
                                            .font(.headline)
                                            .padding(.horizontal)
                                        
                                        VimeoPlayerView(vimeoID: vimeoID, shouldAutoplay: false)
                                            .frame(height: 200)
                                            .cornerRadius(12)
                                            .padding(.horizontal)
                                    }
                                }
                                
                                Spacer(minLength: 100)
                            }
                        }
                        
                        // Bottom Controls
                        BottomControlsView(
                            isTimerRunning: isTimerRunning,
                            isLastStep: isLastStep,
                            timeRemaining: timeRemaining,
                            onPlayPause: toggleTimer,
                            onSkip: nextStep,
                            onFinish: nextStep
                        )
                    }
                }
            }
            .navigationBarHidden(true)
            .onAppear {
                setupCurrentStep()
            }
            .onDisappear {
                stopTimer()
            }
        }
        .navigationViewStyle(StackNavigationViewStyle()) // Prevents issues on iPad
        .sheet(isPresented: $showingPauseMenu) {
            PauseMenuView(
                onResume: { showingPauseMenu = false },
                onRestart: restartSession,
                onExit: { presentationMode.wrappedValue.dismiss() }
            )
        }
    }
    
    // MARK: - Timer Functions
    private func setupCurrentStep() {
        timeRemaining = currentMethod.duration * 60 // Convert minutes to seconds
        isTimerRunning = false
    }
    
    private func startTimer() {
        guard timeRemaining > 0 else { return }
        
        isTimerRunning = true
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            if timeRemaining > 0 {
                timeRemaining -= 1
            } else {
                // Time's up - auto advance to next step
                nextStep()
            }
        }
    }
    
    private func stopTimer() {
        timer?.invalidate()
        timer = nil
        isTimerRunning = false
    }
    
    private func toggleTimer() {
        if isTimerRunning {
            stopTimer()
        } else {
            startTimer()
        }
    }
    
    private func nextStep() {
        stopTimer()
        
        if currentStepIndex < methods.count - 1 {
            currentStepIndex += 1
            setupCurrentStep()
        } else {
            // Session complete
            sessionComplete = true
        }
    }
    
    private func restartSession() {
        stopTimer()
        currentStepIndex = 0
        setupCurrentStep()
        sessionComplete = false
        showingPauseMenu = false
    }
}

// MARK: - Progress Header
struct ProgressHeaderView: View {
    let currentStep: Int
    let totalSteps: Int
    let onPause: () -> Void
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Button(action: onPause) {
                    Image(systemName: "pause.circle")
                        .font(.title2)
                        .foregroundColor(.primary)
                }
                
                Spacer()
                
                VStack(spacing: 4) {
                    Text("Step \(currentStep) of \(totalSteps)")
                        .font(.headline)
                        .fontWeight(.medium)
                }
                
                Spacer()
                
                // Placeholder for symmetry
                Image(systemName: "pause.circle")
                    .font(.title2)
                    .opacity(0)
            }
            
            // Segmented Progress Bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background segments
                    HStack(spacing: 6) {
                        ForEach(0..<totalSteps, id: \.self) { stepIndex in
                            Rectangle()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 8)
                                .cornerRadius(4)
                        }
                    }
                    
                    // Gradient overlay for completed segments
                    if currentStep > 0 {
                        LinearGradient(
                            colors: [Color.brandTeal, Color.brandTealDark],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                        .mask(
                            HStack(spacing: 6) {
                                ForEach(0..<totalSteps, id: \.self) { stepIndex in
                                    let isCompleted = stepIndex < currentStep
                                    let isCurrent = stepIndex == currentStep - 1
                                    
                                    Rectangle()
                                        .fill(isCompleted ? Color.black : (isCurrent ? Color.black.opacity(0.7) : Color.clear))
                                        .frame(height: 8)
                                        .cornerRadius(4)
                                }
                            }
                        )
                        .animation(.easeInOut(duration: 0.3), value: currentStep)
                    }
                }
            }
            .frame(height: 8)
        }
        .padding(.horizontal, 20)
        .padding(.top, 10)
        .padding(.bottom, 8)
    }
}

// MARK: - Timer Circle
struct TimerCircleView: View {
    let timeRemaining: Int
    let totalTime: Int
    let isRunning: Bool
    
    private var progress: Double {
        guard totalTime > 0 else { return 0 }
        return Double(totalTime - timeRemaining) / Double(totalTime)
    }
    
    private var formattedTime: String {
        let minutes = timeRemaining / 60
        let seconds = timeRemaining % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
    
    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .stroke(Color.gray.opacity(0.2), lineWidth: 8)
                .frame(width: 200, height: 200)
            
            // Progress circle
            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    LinearGradient(
                        colors: [Color.brandTeal, Color.brandTealDark],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    style: StrokeStyle(lineWidth: 8, lineCap: .round)
                )
                .frame(width: 200, height: 200)
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 1), value: progress)
            
            // Time display
            VStack(spacing: 8) {
                Text(formattedTime)
                    .font(.system(size: 36, weight: .bold, design: .monospaced))
                    .foregroundColor(.primary)
                
                Text(isRunning ? "In Progress" : "Tap Play")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - Method Details
struct MethodDetailsView: View {
    let method: RecoveryMethod
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Method name and category
            VStack(alignment: .leading, spacing: 8) {
                Text(method.name)
                    .font(.title)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.leading)
                
                Text(method.category)
                    .font(.subheadline)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(Color.brandTeal.opacity(0.2))
                    .cornerRadius(8)
            }
            
            // Description
            Text(method.description)
                .font(.body)
                .lineSpacing(4)
                .foregroundColor(.primary)
            
            // Equipment needed
            if !method.equipment.isEmpty {
                HStack {
                    Image(systemName: "wrench.adjustable")
                        .foregroundColor(.brandTeal)
                    Text("Equipment: \(method.equipment.joined(separator: ", "))")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 8)
            }
            
            // Difficulty indicator
            HStack {
                Text("Difficulty:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                ForEach(1...3, id: \.self) { level in
                    Circle()
                        .fill(level <= method.difficulty ? Color.brandTeal : Color.gray.opacity(0.3))
                        .frame(width: 8, height: 8)
                }
                
                Spacer()
            }
        }
        .padding(.horizontal, 20)
    }
}

// MARK: - Bottom Controls
struct BottomControlsView: View {
    let isTimerRunning: Bool
    let isLastStep: Bool
    let timeRemaining: Int
    let onPlayPause: () -> Void
    let onSkip: () -> Void
    let onFinish: () -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Divider()
            
            HStack(spacing: 20) {
                // Play/Pause Button
                Button(action: onPlayPause) {
                    HStack(spacing: 8) {
                        Image(systemName: isTimerRunning ? "pause.fill" : "play.fill")
                            .font(.title2)
                        Text(isTimerRunning ? "Pause" : "Start")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 16)
                    .background(
                        LinearGradient(
                            colors: [Color.brandTeal, Color.brandTealDark],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(25)
                }
                
                // Skip/Finish Button
                Button(action: isLastStep ? onFinish : onSkip) {
                    HStack(spacing: 8) {
                        Text(isLastStep ? "Finish" : "Skip")
                            .font(.headline)
                        Image(systemName: isLastStep ? "checkmark" : "forward.fill")
                            .font(.title2)
                    }
                    .foregroundColor(.brandTeal)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 16)
                    .background(Color.brandTeal.opacity(0.1))
                    .cornerRadius(25)
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .background(Color(.systemBackground))
    }
}

// MARK: - Pause Menu
struct PauseMenuView: View {
    let onResume: () -> Void
    let onRestart: () -> Void
    let onExit: () -> Void
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                Spacer()
                
                Image(systemName: "pause.circle.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.brandTeal)
                
                Text("Session Paused")
                    .font(.title)
                    .fontWeight(.bold)
                
                Text("What would you like to do?")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                VStack(spacing: 16) {
                    Button(action: {
                        presentationMode.wrappedValue.dismiss()
                        onResume()
                    }) {
                        Text("Resume Session")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.brandTeal)
                            .cornerRadius(12)
                    }
                    
                    Button(action: {
                        presentationMode.wrappedValue.dismiss()
                        onRestart()
                    }) {
                        Text("Restart from Beginning")
                            .font(.headline)
                            .foregroundColor(.brandTeal)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.brandTeal.opacity(0.1))
                            .cornerRadius(12)
                    }
                    
                    Button(action: {
                        presentationMode.wrappedValue.dismiss()
                        onExit()
                    }) {
                        Text("Exit Session")
                            .font(.headline)
                            .foregroundColor(.red)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal, 20)
                
                Spacer()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Session Complete View
struct SessionCompleteView: View {
    let totalMethods: Int
    let totalTime: Int
    let onDismiss: () -> Void
    
    var body: some View {
        VStack(spacing: 32) {
            Spacer()
            
            // Success animation/icon
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.brandTeal.opacity(0.2), Color.brandTeal.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 120, height: 120)
                
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.brandTeal)
            }
            
            VStack(spacing: 12) {
                Text("Recovery Complete!")
                    .font(.title)
                    .fontWeight(.bold)
                
                Text("Great job! You've completed all \(totalMethods) recovery methods.")
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 40)
            }
            
            // Session stats
            VStack(spacing: 16) {
                HStack(spacing: 40) {
                    VStack {
                        Text("\(totalMethods)")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.brandTeal)
                        Text("Methods")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    VStack {
                        Text("\(totalTime)")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.brandTeal)
                        Text("Minutes")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
            }
            
            Spacer()
            
            Button(action: onDismiss) {
                Text("Done")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        LinearGradient(
                            colors: [Color.brandTeal, Color.brandTealDark],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 40)
        }
    }
}
