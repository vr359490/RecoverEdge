//
//  ChatView.swift
//  RecoverEdge
//
//  AI Chat Interface for Recovery Assistance
//

import SwiftUI

// MARK: - Chat Models
struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let timestamp: Date
    
    init(text: String, isUser: Bool) {
        self.text = text
        self.isUser = isUser
        self.timestamp = Date()
    }
}

// MARK: - Chat View
struct ChatView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var messages: [ChatMessage] = []
    @State private var messageText: String = ""
    @State private var isLoading: Bool = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Messages List
                ScrollView {
                    LazyVStack(spacing: 12) {
                        // Welcome message
                        if messages.isEmpty {
                            WelcomeMessageView()
                                .padding(.top, 20)
                        }
                        
                        // Chat messages
                        ForEach(messages) { message in
                            MessageBubble(message: message)
                        }
                        
                        // Loading indicator
                        if isLoading {
                            LoadingMessageBubble()
                        }
                        
                        // Invisible anchor for scrolling
                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 10)
                }
                
                // Message Input
                MessageInputView(
                    messageText: $messageText,
                    isLoading: isLoading,
                    onSend: sendMessage
                )
            }
            .navigationTitle("Recovery Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .background(Color(.systemGroupedBackground))
        }
    }
    
    // Removed scrollToBottom function - using simpler approach
    
    private func sendMessage() {
        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = ChatMessage(text: messageText, isUser: true)
        messages.append(userMessage)
        
        let currentMessage = messageText
        messageText = ""
        isLoading = true
        
        // TODO: Replace this with your actual OpenAI API call
        sendToAI(message: currentMessage)
    }
    
    private func sendToAI(message: String) {
        // Simulate API call for now - replace with your OpenAI implementation
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            let aiResponse = generateMockResponse(for: message)
            let aiMessage = ChatMessage(text: aiResponse, isUser: false)
            
            withAnimation(.easeInOut) {
                isLoading = false
                messages.append(aiMessage)
            }
        }
    }
    
    // Mock response generator - replace with actual AI
    private func generateMockResponse(for message: String) -> String {
        let lowercaseMessage = message.lowercased()
        
        if lowercaseMessage.contains("foam roll") {
            return "Foam rolling is excellent for recovery! It helps break up fascial adhesions and improve blood flow. I recommend 30-60 seconds per muscle group, focusing on areas that feel tight. Would you like me to suggest a specific foam rolling routine?"
        } else if lowercaseMessage.contains("stretch") {
            return "Stretching is crucial for recovery and injury prevention. Static stretches work best post-workout when muscles are warm. Hold each stretch for 20-30 seconds. What specific areas would you like to stretch?"
        } else if lowercaseMessage.contains("sleep") {
            return "Sleep is when most recovery happens! Aim for 7-9 hours per night. Your body releases growth hormone during deep sleep, which repairs muscle tissue. Creating a consistent bedtime routine can improve sleep quality."
        } else if lowercaseMessage.contains("pain") || lowercaseMessage.contains("sore") {
            return "Muscle soreness is normal after exercise, but persistent pain isn't. For soreness, try gentle movement, light stretching, or foam rolling. If you have sharp or persistent pain, please consult a healthcare professional."
        } else {
            return "That's a great question about recovery! I'm here to help you optimize your recovery routine. Feel free to ask about specific techniques, equipment, or recovery strategies. What aspect of recovery would you like to explore?"
        }
    }
}

// MARK: - Welcome Message
struct WelcomeMessageView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 50))
                .foregroundColor(.brandTeal)
            
            Text("Recovery Assistant")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Ask me anything about recovery techniques, equipment, or methods. I'm here to help optimize your recovery routine!")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal, 20)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Try asking:")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                SuggestedQuestion(text: "How do I foam roll my IT band?")
                SuggestedQuestion(text: "What's the best recovery for sore legs?")
                SuggestedQuestion(text: "How long should I stretch after a workout?")
            }
            .padding(.top, 10)
        }
        .padding()
    }
}

struct SuggestedQuestion: View {
    let text: String
    
    var body: some View {
        Text(text)
            .font(.caption)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.brandTeal.opacity(0.1))
            .cornerRadius(12)
    }
}

// MARK: - Message Bubble
struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer(minLength: 60)
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(message.text)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(
                            LinearGradient(
                                colors: [Color.brandTeal, Color.brandTealDark],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .foregroundColor(.white)
                        .cornerRadius(18)
                    
                    Text(formatTime(message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.trailing, 8)
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "brain.head.profile")
                            .font(.caption)
                            .foregroundColor(.brandTeal)
                            .padding(.top, 2)
                        
                        Text(message.text)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(Color(.systemGray5))
                            .foregroundColor(.primary)
                            .cornerRadius(18)
                    }
                    
                    Text(formatTime(message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.leading, 28)
                }
                
                Spacer(minLength: 60)
            }
        }
    }
    
    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Loading Message
struct LoadingMessageBubble: View {
    @State private var animationPhase = 0
    
    var body: some View {
        HStack {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "brain.head.profile")
                    .font(.caption)
                    .foregroundColor(.brandTeal)
                    .padding(.top, 2)
                
                HStack(spacing: 4) {
                    ForEach(0..<3, id: \.self) { index in
                        Circle()
                            .fill(Color.secondary)
                            .frame(width: 8, height: 8)
                            .scaleEffect(animationPhase == index ? 1.2 : 0.8)
                            .animation(
                                Animation.easeInOut(duration: 0.6)
                                    .repeatForever()
                                    .delay(Double(index) * 0.2),
                                value: animationPhase
                            )
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color(.systemGray5))
                .cornerRadius(18)
            }
            
            Spacer(minLength: 60)
        }
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.6, repeats: true) { _ in
                animationPhase = (animationPhase + 1) % 3
            }
        }
    }
}

// MARK: - Message Input
struct MessageInputView: View {
    @Binding var messageText: String
    let isLoading: Bool
    let onSend: () -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            Divider()
            
            HStack(spacing: 12) {
                TextField("Ask about recovery...", text: $messageText, axis: .vertical)
                    .textFieldStyle(PlainTextFieldStyle())
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color(.systemGray6))
                    .cornerRadius(20)
                    .lineLimit(1...4)
                    .disabled(isLoading)
                
                Button(action: onSend) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 30))
                        .foregroundColor(canSend ? .brandTeal : .gray)
                }
                .disabled(!canSend || isLoading)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(.systemBackground))
        }
    }
    
    private var canSend: Bool {
        !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
