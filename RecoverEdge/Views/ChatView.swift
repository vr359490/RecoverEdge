import SwiftUI

// MARK: - OpenAI API Models
struct OpenAIRequest: Codable {
    let model: String
    let messages: [OpenAIMessage]
    let temperature: Double
    let max_tokens: Int
}

struct OpenAIMessage: Codable {
    let role: String
    let content: String
}

struct OpenAIResponse: Codable {
    let choices: [Choice]
    
    struct Choice: Codable {
        let message: OpenAIMessage
    }
}

// MARK: - OpenAI Service
class OpenAIService {
    static let shared = OpenAIService()
    
    // TODO: Replace with your actual API key - NEVER commit this to source control!
    // Consider using environment variables or secure storage
    private let apiKey = "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"
    private let apiURL = "https://api.openai.com/v1/chat/completions"
    
    private init() {}
    
    func sendMessage(_ message: String, conversationHistory: [ChatMessage]) async throws -> String {
        guard let url = URL(string: apiURL) else {
            throw URLError(.badURL)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Build conversation history for context
        var messages: [OpenAIMessage] = [
            OpenAIMessage(
                role: "system",
                content: """
                You are a knowledgeable recovery and fitness assistant specializing in post-workout recovery techniques.
                You provide evidence-based advice on recovery methods including foam rolling, stretching, cold therapy,
                breathing techniques, and proper rest. Keep responses helpful, concise, and focused on practical recovery advice.
                When discussing specific techniques, provide clear instructions and explain the benefits.
                If someone mentions pain or injury, remind them to consult healthcare professionals for medical advice.
                """
            )
        ]
        
        // Add conversation history (limit to last 10 messages to manage token usage)
        let recentHistory = conversationHistory.suffix(10)
        for historyMessage in recentHistory {
            messages.append(OpenAIMessage(
                role: historyMessage.isUser ? "user" : "assistant",
                content: historyMessage.text
            ))
        }
        
        // Add current message
        messages.append(OpenAIMessage(role: "user", content: message))
        
        let requestBody = OpenAIRequest(
            model: "gpt-3.5-turbo", // or "gpt-4" if you have access
            messages: messages,
            temperature: 0.7,
            max_tokens: 500
        )
        
        request.httpBody = try JSONEncoder().encode(requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        
        if httpResponse.statusCode != 200 {
            // Try to parse error message
            if let errorData = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let error = errorData["error"] as? [String: Any],
               let message = error["message"] as? String {
                throw NSError(domain: "OpenAI", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: message])
            }
            throw URLError(.badServerResponse)
        }
        
        let openAIResponse = try JSONDecoder().decode(OpenAIResponse.self, from: data)
        
        guard let firstChoice = openAIResponse.choices.first else {
            throw NSError(domain: "OpenAI", code: 0, userInfo: [NSLocalizedDescriptionKey: "No response from AI"])
        }
        
        return firstChoice.message.content
    }
}

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
    @State private var showingError: Bool = false
    @State private var errorMessage: String = ""
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Messages List
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            // Welcome message
                            if messages.isEmpty {
                                WelcomeMessageView(onSuggestionTap: { suggestion in
                                    messageText = suggestion
                                    sendMessage()
                                })
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
                    .onChange(of: messages.count) { _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
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
            .alert("Error", isPresented: $showingError) {
                Button("OK") {
                    showingError = false
                }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func sendMessage() {
        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = ChatMessage(text: messageText, isUser: true)
        messages.append(userMessage)
        
        let currentMessage = messageText
        messageText = ""
        isLoading = true
        
        Task {
            await sendToOpenAI(message: currentMessage)
        }
    }
    
    private func sendToOpenAI(message: String) async {
        do {
            let response = try await OpenAIService.shared.sendMessage(message, conversationHistory: messages)
            
            await MainActor.run {
                let aiMessage = ChatMessage(text: response, isUser: false)
                withAnimation(.easeInOut) {
                    isLoading = false
                    messages.append(aiMessage)
                }
            }
        } catch {
            await MainActor.run {
                isLoading = false
                errorMessage = "Failed to get response: \(error.localizedDescription)"
                showingError = true
                
                // Add a fallback message
                let fallbackMessage = ChatMessage(
                    text: "I apologize, but I'm having trouble connecting right now. Please check your internet connection and try again.",
                    isUser: false
                )
                messages.append(fallbackMessage)
            }
        }
    }
}

// MARK: - Welcome Message
struct WelcomeMessageView: View {
    let onSuggestionTap: (String) -> Void
    
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
                
                SuggestedQuestion(text: "How do I foam roll my IT band?") {
                    onSuggestionTap("How do I foam roll my IT band?")
                }
                SuggestedQuestion(text: "What's the best recovery for sore legs?") {
                    onSuggestionTap("What's the best recovery for sore legs?")
                }
                SuggestedQuestion(text: "How long should I stretch after a workout?") {
                    onSuggestionTap("How long should I stretch after a workout?")
                }
            }
            .padding(.top, 10)
        }
        .padding()
    }
}

struct SuggestedQuestion: View {
    let text: String
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            Text(text)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.brandTeal.opacity(0.1))
                .cornerRadius(12)
                .foregroundColor(.primary)
        }
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
    @FocusState private var isTextFieldFocused: Bool
    
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
                    .focused($isTextFieldFocused)
                    .onSubmit {
                        if canSend {
                            onSend()
                        }
                    }
                
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

////
////  ChatView.swift
////  RecoverEdge
////
////  AI Chat Interface for Recovery Assistance
////
//
//import SwiftUI
//
//// MARK: - Chat Models
//struct ChatMessage: Identifiable, Equatable {
//    let id = UUID()
//    let text: String
//    let isUser: Bool
//    let timestamp: Date
//
//    init(text: String, isUser: Bool) {
//        self.text = text
//        self.isUser = isUser
//        self.timestamp = Date()
//    }
//}
//
//// MARK: - Chat View
//struct ChatView: View {
//    @EnvironmentObject var dataStore: RecoveryDataStore
//    @State private var messages: [ChatMessage] = []
//    @State private var messageText: String = ""
//    @State private var isLoading: Bool = false
//
//    var body: some View {
//        NavigationView {
//            VStack(spacing: 0) {
//                // Messages List
//                ScrollView {
//                    LazyVStack(spacing: 12) {
//                        // Welcome message
//                        if messages.isEmpty {
//                            WelcomeMessageView()
//                                .padding(.top, 20)
//                        }
//
//                        // Chat messages
//                        ForEach(messages) { message in
//                            MessageBubble(message: message)
//                        }
//
//                        // Loading indicator
//                        if isLoading {
//                            LoadingMessageBubble()
//                        }
//
//                        // Invisible anchor for scrolling
//                        Color.clear
//                            .frame(height: 1)
//                            .id("bottom")
//                    }
//                    .padding(.horizontal, 16)
//                    .padding(.bottom, 10)
//                }
//
//                // Message Input
//                MessageInputView(
//                    messageText: $messageText,
//                    isLoading: isLoading,
//                    onSend: sendMessage
//                )
//            }
//            .navigationTitle("Recovery Assistant")
//            .navigationBarTitleDisplayMode(.inline)
//            .background(Color(.systemGroupedBackground))
//        }
//    }
//
//    // Removed scrollToBottom function - using simpler approach
//
//    private func sendMessage() {
//        guard !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
//
//        let userMessage = ChatMessage(text: messageText, isUser: true)
//        messages.append(userMessage)
//
//        let currentMessage = messageText
//        messageText = ""
//        isLoading = true
//
//        // TODO: Replace this with your actual OpenAI API call
//        sendToAI(message: currentMessage)
//    }
//
//    private func sendToAI(message: String) {
//        // Simulate API call for now - replace with your OpenAI implementation
//        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
//            let aiResponse = generateMockResponse(for: message)
//            let aiMessage = ChatMessage(text: aiResponse, isUser: false)
//
//            withAnimation(.easeInOut) {
//                isLoading = false
//                messages.append(aiMessage)
//            }
//        }
//    }
//
//    // Mock response generator - replace with actual AI
//    private func generateMockResponse(for message: String) -> String {
//        let lowercaseMessage = message.lowercased()
//
//        if lowercaseMessage.contains("foam roll") {
//            return "Foam rolling is excellent for recovery! It helps break up fascial adhesions and improve blood flow. I recommend 30-60 seconds per muscle group, focusing on areas that feel tight. Would you like me to suggest a specific foam rolling routine?"
//        } else if lowercaseMessage.contains("stretch") {
//            return "Stretching is crucial for recovery and injury prevention. Static stretches work best post-workout when muscles are warm. Hold each stretch for 20-30 seconds. What specific areas would you like to stretch?"
//        } else if lowercaseMessage.contains("sleep") {
//            return "Sleep is when most recovery happens! Aim for 7-9 hours per night. Your body releases growth hormone during deep sleep, which repairs muscle tissue. Creating a consistent bedtime routine can improve sleep quality."
//        } else if lowercaseMessage.contains("pain") || lowercaseMessage.contains("sore") {
//            return "Muscle soreness is normal after exercise, but persistent pain isn't. For soreness, try gentle movement, light stretching, or foam rolling. If you have sharp or persistent pain, please consult a healthcare professional."
//        } else {
//            return "That's a great question about recovery! I'm here to help you optimize your recovery routine. Feel free to ask about specific techniques, equipment, or recovery strategies. What aspect of recovery would you like to explore?"
//        }
//    }
//}
//
//// MARK: - Welcome Message
//struct WelcomeMessageView: View {
//    var body: some View {
//        VStack(spacing: 16) {
//            Image(systemName: "brain.head.profile")
//                .font(.system(size: 50))
//                .foregroundColor(.brandTeal)
//
//            Text("Recovery Assistant")
//                .font(.title2)
//                .fontWeight(.bold)
//
//            Text("Ask me anything about recovery techniques, equipment, or methods. I'm here to help optimize your recovery routine!")
//                .multilineTextAlignment(.center)
//                .foregroundColor(.secondary)
//                .padding(.horizontal, 20)
//
//            VStack(alignment: .leading, spacing: 8) {
//                Text("Try asking:")
//                    .font(.caption)
//                    .fontWeight(.medium)
//                    .foregroundColor(.secondary)
//
//                SuggestedQuestion(text: "How do I foam roll my IT band?")
//                SuggestedQuestion(text: "What's the best recovery for sore legs?")
//                SuggestedQuestion(text: "How long should I stretch after a workout?")
//            }
//            .padding(.top, 10)
//        }
//        .padding()
//    }
//}
//
//struct SuggestedQuestion: View {
//    let text: String
//
//    var body: some View {
//        Text(text)
//            .font(.caption)
//            .padding(.horizontal, 12)
//            .padding(.vertical, 6)
//            .background(Color.brandTeal.opacity(0.1))
//            .cornerRadius(12)
//    }
//}
//
//// MARK: - Message Bubble
//struct MessageBubble: View {
//    let message: ChatMessage
//
//    var body: some View {
//        HStack {
//            if message.isUser {
//                Spacer(minLength: 60)
//
//                VStack(alignment: .trailing, spacing: 4) {
//                    Text(message.text)
//                        .padding(.horizontal, 16)
//                        .padding(.vertical, 12)
//                        .background(
//                            LinearGradient(
//                                colors: [Color.brandTeal, Color.brandTealDark],
//                                startPoint: .topLeading,
//                                endPoint: .bottomTrailing
//                            )
//                        )
//                        .foregroundColor(.white)
//                        .cornerRadius(18)
//
//                    Text(formatTime(message.timestamp))
//                        .font(.caption2)
//                        .foregroundColor(.secondary)
//                        .padding(.trailing, 8)
//                }
//            } else {
//                VStack(alignment: .leading, spacing: 4) {
//                    HStack(alignment: .top, spacing: 8) {
//                        Image(systemName: "brain.head.profile")
//                            .font(.caption)
//                            .foregroundColor(.brandTeal)
//                            .padding(.top, 2)
//
//                        Text(message.text)
//                            .padding(.horizontal, 16)
//                            .padding(.vertical, 12)
//                            .background(Color(.systemGray5))
//                            .foregroundColor(.primary)
//                            .cornerRadius(18)
//                    }
//
//                    Text(formatTime(message.timestamp))
//                        .font(.caption2)
//                        .foregroundColor(.secondary)
//                        .padding(.leading, 28)
//                }
//
//                Spacer(minLength: 60)
//            }
//        }
//    }
//
//    private func formatTime(_ date: Date) -> String {
//        let formatter = DateFormatter()
//        formatter.timeStyle = .short
//        return formatter.string(from: date)
//    }
//}
//
//// MARK: - Loading Message
//struct LoadingMessageBubble: View {
//    @State private var animationPhase = 0
//
//    var body: some View {
//        HStack {
//            HStack(alignment: .top, spacing: 8) {
//                Image(systemName: "brain.head.profile")
//                    .font(.caption)
//                    .foregroundColor(.brandTeal)
//                    .padding(.top, 2)
//
//                HStack(spacing: 4) {
//                    ForEach(0..<3, id: \.self) { index in
//                        Circle()
//                            .fill(Color.secondary)
//                            .frame(width: 8, height: 8)
//                            .scaleEffect(animationPhase == index ? 1.2 : 0.8)
//                            .animation(
//                                Animation.easeInOut(duration: 0.6)
//                                    .repeatForever()
//                                    .delay(Double(index) * 0.2),
//                                value: animationPhase
//                            )
//                    }
//                }
//                .padding(.horizontal, 16)
//                .padding(.vertical, 12)
//                .background(Color(.systemGray5))
//                .cornerRadius(18)
//            }
//
//            Spacer(minLength: 60)
//        }
//        .onAppear {
//            Timer.scheduledTimer(withTimeInterval: 0.6, repeats: true) { _ in
//                animationPhase = (animationPhase + 1) % 3
//            }
//        }
//    }
//}
//
//// MARK: - Message Input
//struct MessageInputView: View {
//    @Binding var messageText: String
//    let isLoading: Bool
//    let onSend: () -> Void
//
//    var body: some View {
//        VStack(spacing: 0) {
//            Divider()
//
//            HStack(spacing: 12) {
//                TextField("Ask about recovery...", text: $messageText, axis: .vertical)
//                    .textFieldStyle(PlainTextFieldStyle())
//                    .padding(.horizontal, 16)
//                    .padding(.vertical, 12)
//                    .background(Color(.systemGray6))
//                    .cornerRadius(20)
//                    .lineLimit(1...4)
//                    .disabled(isLoading)
//
//                Button(action: onSend) {
//                    Image(systemName: "arrow.up.circle.fill")
//                        .font(.system(size: 30))
//                        .foregroundColor(canSend ? .brandTeal : .gray)
//                }
//                .disabled(!canSend || isLoading)
//            }
//            .padding(.horizontal, 16)
//            .padding(.vertical, 12)
//            .background(Color(.systemBackground))
//        }
//    }
//
//    private var canSend: Bool {
//        !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
//    }
//}


//
//  ChatView.swift
//  RecoverEdge
//
//  AI Chat Interface for Recovery Assistance
//


