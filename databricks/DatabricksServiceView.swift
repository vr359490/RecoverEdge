//
//  DatabricksService.swift
//  RecoverEdge
//
//  Enhanced service for Databricks integration with analytics and recommendations
//

import SwiftUI
import Foundation
import Combine

// MARK: - Databricks API Models
struct DatabricksResponse<T: Codable>: Codable {
    let data: T?
    let error: String?
    let poweredBy: String?
    let retrievedChunks: Int?
}

struct RecommendationRequest: Codable {
    let userId: String
    let userFeatures: UserFeatures
    let context: RecommendationContext
}

struct UserFeatures: Codable {
    var totalInteractions: Int
    var uniqueSessions: Int
    var avgSessionDuration: Double
    var avgSatisfaction: Double
    let fitnessLevel: String
    let ageGroup: String
    let preferredLocations: [String]
    let equipmentUsed: [String]
    var completionEfficiency: Double
    let avgMethodEffectiveness: Double
}

struct RecommendationContext: Codable {
    let currentLocation: String
    let availableTime: Int
    let availableEquipment: [String]
    let previousMethods: [String]
    let timeOfDay: String
}

struct MethodRecommendation: Codable, Identifiable {
    let id = UUID()
    let method: String
    let confidence: Double
    let reason: String
    let estimatedDuration: Int?
    let requiredEquipment: [String]?
    
    private enum CodingKeys: String, CodingKey {
        case method, confidence, reason, estimatedDuration, requiredEquipment
    }
}

struct RecommendationResponse: Codable {
    let recommendedMethods: [MethodRecommendation]
    let optimalDuration: Int
    let preferredLocation: String
    let modelVersion: String
    let effectivenessScore: Double
    let engagementProbability: Double
}

struct AnalyticsInsight: Codable {
    let popularTopics: [PopularTopic]
    let userEngagement: UserEngagement
    let methodEffectiveness: MethodEffectiveness
}

struct PopularTopic: Codable, Identifiable {
    let id = UUID()
    let topic: String
    let queryCount: Int
    let avgRating: Double
    
    private enum CodingKeys: String, CodingKey {
        case topic, queryCount, avgRating
    }
}

struct UserEngagement: Codable {
    let dailyActiveUsers: Int
    let avgSessionDuration: Double
    let completionRate: Double
}

struct MethodEffectiveness: Codable {
    let topRated: String
    let mostCompleted: String
    let avgSatisfaction: Double
}

struct InteractionEvent: Codable {
    let eventType: String // "chat", "plan_generation", "session_start", "session_complete"
    let timestamp: Date
    let userId: String
    let sessionId: String
    let metadata: String // Changed from [String: Any] to String for Codable compliance
    
    init(eventType: String, timestamp: Date, userId: String, sessionId: String, metadata: [String: Any]) {
        self.eventType = eventType
        self.timestamp = timestamp
        self.userId = userId
        self.sessionId = sessionId
        
        // Convert metadata dictionary to JSON string
        if let jsonData = try? JSONSerialization.data(withJSONObject: metadata),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            self.metadata = jsonString
        } else {
            self.metadata = "{}"
        }
    }
}

// MARK: - Enhanced Databricks Service
class DatabricksService: ObservableObject {
    static let shared = DatabricksService()
    
    private let baseURL: String
    private let apiKey: String
    private var cancellables = Set<AnyCancellable>()
    
    // User identification
    @Published var currentUserId: String
    @Published var currentSessionId: String
    
    // Analytics tracking
    private var userFeatures: UserFeatures?
    private var sessionStartTime: Date?
    private var interactionCount: Int = 0
    
    private init() {
        self.baseURL = "http://localhost:8000" // Update with your Databricks server URL
        self.apiKey = "" // Add authentication if needed
        
        // Generate or retrieve user ID
        let userId = UserDefaults.standard.string(forKey: "databricks_user_id") ?? UUID().uuidString
        self.currentUserId = userId
        UserDefaults.standard.set(userId, forKey: "databricks_user_id")
        
        self.currentSessionId = UUID().uuidString
        
        // Load user features
        self.loadUserFeatures()
    }
    
    // MARK: - Chat with Analytics
    func sendMessage(_ message: String, conversationHistory: [ChatMessage]) async throws -> String {
        let url = URL(string: "\(baseURL)/chat")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = [
            "message": message,
            "user_id": currentUserId,
            "session_id": currentSessionId,
            "conversationHistory": conversationHistory.map { chatMessage in
                [
                    "role": chatMessage.isUser ? "user" : "assistant",
                    "content": chatMessage.text
                ]
            }
        ] as [String: Any]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        // Parse response
        if let responseDict = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let responseText = responseDict["response"] as? String {
            
            // Track interaction
            await trackInteraction(
                eventType: "chat",
                metadata: [
                    "message_length": message.count,
                    "response_length": responseText.count,
                    "retrieved_chunks": responseDict["retrievedChunks"] as? Int ?? 0
                ]
            )
            
            return responseText
        } else {
            throw NSError(domain: "DatabricksService", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
        }
    }
    
    // MARK: - Personalized Recommendations
    func getPersonalizedRecommendations(
        location: Location,
        availableTime: Int,
        availableEquipment: [String]
    ) async throws -> RecommendationResponse {
        
        let url = URL(string: "\(baseURL)/recommendations/\(currentUserId)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Build recommendation context
        let context = RecommendationContext(
            currentLocation: location.rawValue,
            availableTime: availableTime,
            availableEquipment: availableEquipment,
            previousMethods: getPreviousMethods(),
            timeOfDay: getCurrentTimeOfDay()
        )
        
        let requestBody = RecommendationRequest(
            userId: currentUserId,
            userFeatures: userFeatures ?? getDefaultUserFeatures(),
            context: context
        )
        
        request.httpBody = try JSONEncoder().encode(requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        let recommendations = try JSONDecoder().decode(RecommendationResponse.self, from: data)
        
        // Track recommendation request
        await trackInteraction(
            eventType: "recommendation_request",
            metadata: [
                "location": location.rawValue,
                "available_time": availableTime,
                "equipment_count": availableEquipment.count,
                "recommended_methods": recommendations.recommendedMethods.count
            ]
        )
        
        return recommendations
    }
    
    // MARK: - Analytics and Insights
    func getAnalyticsInsights() async throws -> AnalyticsInsight {
        let url = URL(string: "\(baseURL)/analytics/insights")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return try JSONDecoder().decode(AnalyticsInsight.self, from: data)
    }
    
    // MARK: - Session Tracking
    func startRecoverySession(methods: [RecoveryMethod], plannedDuration: Int) {
        sessionStartTime = Date()
        
        Task {
            await trackInteraction(
                eventType: "session_start",
                metadata: [
                    "planned_duration": plannedDuration,
                    "method_count": methods.count,
                    "methods": methods.map { $0.name }
                ]
            )
        }
    }
    
    func completeRecoverySession(
        completedMethods: [String],
        skippedMethods: [String],
        actualDuration: Int,
        satisfactionRating: Int?
    ) {
        Task {
            await trackInteraction(
                eventType: "session_complete",
                metadata: [
                    "actual_duration": actualDuration,
                    "completed_methods": completedMethods,
                    "skipped_methods": skippedMethods,
                    "satisfaction_rating": satisfactionRating ?? -1,
                    "completion_rate": Double(completedMethods.count) / Double(completedMethods.count + skippedMethods.count)
                ]
            )
        }
        
        // Update user features
        updateUserFeatures(
            sessionDuration: actualDuration,
            completionRate: Double(completedMethods.count) / Double(completedMethods.count + skippedMethods.count),
            satisfaction: satisfactionRating
        )
        
        // Start new session
        currentSessionId = UUID().uuidString
    }
    
    func rateMethod(_ methodName: String, rating: Int, benefits: [String] = []) {
        Task {
            await trackInteraction(
                eventType: "method_rating",
                metadata: [
                    "method_name": methodName,
                    "rating": rating,
                    "benefits": benefits
                ]
            )
        }
    }
    
    // MARK: - Private Helper Methods
    private func trackInteraction(eventType: String, metadata: [String: Any]) async {
        do {
            let event = InteractionEvent(
                eventType: eventType,
                timestamp: Date(),
                userId: currentUserId,
                sessionId: currentSessionId,
                metadata: metadata
            )
            
            let url = URL(string: "\(baseURL)/analytics/track")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            request.httpBody = try JSONEncoder().encode(event)
            
            let (_, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode != 200 {
                print("Failed to track interaction: \(httpResponse.statusCode)")
            }
            
            interactionCount += 1
            
        } catch {
            print("Error tracking interaction: \(error)")
        }
    }
    
    private func loadUserFeatures() {
        if let data = UserDefaults.standard.data(forKey: "user_features"),
           let features = try? JSONDecoder().decode(UserFeatures.self, from: data) {
            self.userFeatures = features
        } else {
            self.userFeatures = getDefaultUserFeatures()
        }
    }
    
    private func updateUserFeatures(sessionDuration: Int, completionRate: Double, satisfaction: Int?) {
        guard var features = userFeatures else { return }
        
        // Update running averages
        let sessionCount = features.uniqueSessions + 1
        features.avgSessionDuration = ((features.avgSessionDuration * Double(features.uniqueSessions)) + Double(sessionDuration)) / Double(sessionCount)
        features.completionEfficiency = ((features.completionEfficiency * Double(features.uniqueSessions)) + completionRate) / Double(sessionCount)
        
        if let satisfaction = satisfaction {
            features.avgSatisfaction = ((features.avgSatisfaction * Double(features.uniqueSessions)) + Double(satisfaction)) / Double(sessionCount)
        }
        
        features.uniqueSessions = sessionCount
        features.totalInteractions = interactionCount
        
        // Save updated features
        if let data = try? JSONEncoder().encode(features) {
            UserDefaults.standard.set(data, forKey: "user_features")
        }
        
        self.userFeatures = features
    }
    
    private func getDefaultUserFeatures() -> UserFeatures {
        return UserFeatures(
            totalInteractions: 0,
            uniqueSessions: 0,
            avgSessionDuration: 0.0,
            avgSatisfaction: 3.0,
            fitnessLevel: "beginner",
            ageGroup: "unknown",
            preferredLocations: [],
            equipmentUsed: [],
            completionEfficiency: 0.0,
            avgMethodEffectiveness: 3.0
        )
    }
    
    private func getPreviousMethods() -> [String] {
        // Get recently used methods from UserDefaults or local storage
        return UserDefaults.standard.stringArray(forKey: "recent_methods") ?? []
    }
    
    private func getCurrentTimeOfDay() -> String {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 6..<12: return "morning"
        case 12..<17: return "afternoon"
        case 17..<21: return "evening"
        default: return "night"
        }
    }
}

// MARK: - Enhanced Chat View with Databricks Integration
struct EnhancedChatView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @StateObject private var databricksService = DatabricksService.shared
    @State private var messages: [ChatMessage] = []
    @State private var messageText: String = ""
    @State private var isLoading: Bool = false
    @State private var showingError: Bool = false
    @State private var errorMessage: String = ""
    @State private var showingInsights: Bool = false
    @State private var analyticsInsights: AnalyticsInsight?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Enhanced Header with Analytics Button
                HStack {
                    Text("AI Recovery Assistant")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Spacer()
                    
                    Button(action: { showingInsights = true }) {
                        Image(systemName: "chart.bar.fill")
                            .foregroundColor(.brandTeal)
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
                
                // Messages List with Enhanced UI
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            // Welcome message with personalization
                            if messages.isEmpty {
                                EnhancedWelcomeMessageView(
                                    userId: databricksService.currentUserId,
                                    onSuggestionTap: { suggestion in
                                        messageText = suggestion
                                        sendMessage()
                                    }
                                )
                                .padding(.top, 20)
                            }
                            
                            // Chat messages with analytics
                            ForEach(messages) { message in
                                EnhancedMessageBubble(message: message)
                            }
                            
                            // Loading indicator
                            if isLoading {
                                LoadingMessageBubble()
                            }
                            
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
                
                // Enhanced Message Input
                EnhancedMessageInputView(
                    messageText: $messageText,
                    isLoading: isLoading,
                    onSend: sendMessage
                )
            }
            .navigationBarHidden(true)
            .alert("Error", isPresented: $showingError) {
                Button("OK") { showingError = false }
            } message: {
                Text(errorMessage)
            }
            .sheet(isPresented: $showingInsights) {
                AnalyticsInsightsView(insights: analyticsInsights)
            }
        }
        .onAppear {
            loadAnalyticsInsights()
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
            do {
                let response = try await databricksService.sendMessage(currentMessage, conversationHistory: messages)
                
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
                    
                    let fallbackMessage = ChatMessage(
                        text: "I'm having trouble connecting right now. Please try again.",
                        isUser: false
                    )
                    messages.append(fallbackMessage)
                }
            }
        }
    }
    
    private func loadAnalyticsInsights() {
        Task {
            do {
                let insights = try await databricksService.getAnalyticsInsights()
                await MainActor.run {
                    analyticsInsights = insights
                }
            } catch {
                print("Failed to load analytics insights: \(error)")
            }
        }
    }
}

// MARK: - Enhanced Welcome Message with Personalization
struct EnhancedWelcomeMessageView: View {
    let userId: String
    let onSuggestionTap: (String) -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 50))
                .foregroundColor(.brandTeal)
            
            Text("AI Recovery Assistant")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Powered by advanced analytics and personalized recommendations. Ask me anything about recovery!")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
                .padding(.horizontal, 20)
            
            // Personalized suggestions
            VStack(alignment: .leading, spacing: 8) {
                Text("Personalized suggestions:")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                PersonalizedSuggestionButton(
                    text: "What's the best recovery for my fitness level?",
                    onTap: { onSuggestionTap("What's the best recovery for my fitness level?") }
                )
                
                PersonalizedSuggestionButton(
                    text: "Recommend a recovery plan for today",
                    onTap: { onSuggestionTap("Recommend a recovery plan for today") }
                )
                
                PersonalizedSuggestionButton(
                    text: "How can I improve my recovery consistency?",
                    onTap: { onSuggestionTap("How can I improve my recovery consistency?") }
                )
            }
            .padding(.top, 10)
            
            // User ID display (for debugging)
            Text("User ID: \(String(userId.prefix(8)))...")
                .font(.caption2)
                .foregroundColor(.secondary.opacity(0.6))
        }
        .padding()
    }
}

struct PersonalizedSuggestionButton: View {
    let text: String
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack {
                Image(systemName: "sparkles")
                    .foregroundColor(.brandTeal)
                    .font(.caption)
                
                Text(text)
                    .font(.caption)
                
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                LinearGradient(
                    colors: [Color.brandTeal.opacity(0.1), Color.brandTeal.opacity(0.05)],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .cornerRadius(12)
            .foregroundColor(.primary)
        }
    }
}

// MARK: - Enhanced Message Bubble with Analytics
struct EnhancedMessageBubble: View {
    let message: ChatMessage
    @State private var showingAnalytics = false
    
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
                    
                    HStack(spacing: 8) {
                        Text(formatTime(message.timestamp))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        
                        Button(action: { showingAnalytics.toggle() }) {
                            Image(systemName: "chart.line.uptrend.xyaxis")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.trailing, 8)
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "brain.head.profile")
                            .font(.caption)
                            .foregroundColor(.brandTeal)
                            .padding(.top, 2)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text(message.text)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 12)
                                .background(Color(.systemGray5))
                                .foregroundColor(.primary)
                                .cornerRadius(18)
                            
                            // Analytics indicator for AI responses
                            if showingAnalytics {
                                HStack(spacing: 4) {
                                    Image(systemName: "cpu")
                                        .font(.caption2)
                                        .foregroundColor(.brandTeal)
                                    Text("Powered by Databricks ML")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                .padding(.horizontal, 16)
                                .transition(.opacity)
                            }
                        }
                    }
                    
                    HStack(spacing: 8) {
                        Text(formatTime(message.timestamp))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .padding(.leading, 28)
                        
                        Button(action: { withAnimation { showingAnalytics.toggle() } }) {
                            Image(systemName: showingAnalytics ? "info.circle.fill" : "info.circle")
                                .font(.caption2)
                                .foregroundColor(.brandTeal)
                        }
                    }
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

// MARK: - Enhanced Message Input with Smart Suggestions
struct EnhancedMessageInputView: View {
    @Binding var messageText: String
    let isLoading: Bool
    let onSend: () -> Void
    @FocusState private var isTextFieldFocused: Bool
    @State private var smartSuggestions: [String] = []
    
    var body: some View {
        VStack(spacing: 8) {
            // Smart suggestions
            if !smartSuggestions.isEmpty && isTextFieldFocused {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(smartSuggestions, id: \.self) { suggestion in
                            Button(suggestion) {
                                messageText = suggestion
                                smartSuggestions = []
                                onSend()
                            }
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.brandTeal.opacity(0.1))
                            .cornerRadius(12)
                            .foregroundColor(.brandTeal)
                        }
                    }
                    .padding(.horizontal, 16)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            
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
                        .onChange(of: messageText) { newValue in
                            updateSmartSuggestions(for: newValue)
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
        .animation(.easeInOut(duration: 0.3), value: smartSuggestions.isEmpty)
    }
    
    private var canSend: Bool {
        !messageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
    
    private func updateSmartSuggestions(for text: String) {
        let lowercaseText = text.lowercased()
        
        if lowercaseText.contains("foam") {
            smartSuggestions = ["How to foam roll properly?", "Foam rolling for IT band", "Best foam rolling routine"]
        } else if lowercaseText.contains("cold") {
            smartSuggestions = ["Cold therapy benefits", "Ice bath duration", "Cold shower protocol"]
        } else if lowercaseText.contains("stretch") {
            smartSuggestions = ["Post-workout stretching", "Best stretches for legs", "How long to stretch"]
        } else if lowercaseText.contains("sore") {
            smartSuggestions = ["Recovery for sore muscles", "Reduce muscle soreness", "DOMS treatment"]
        } else if text.isEmpty {
            smartSuggestions = []
        } else if text.count > 3 {
            smartSuggestions = ["Get personalized recommendations", "What's trending in recovery?"]
        }
    }
}

// MARK: - Analytics Insights View
struct AnalyticsInsightsView: View {
    let insights: AnalyticsInsight?
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    if let insights = insights {
                        // Popular Topics Section
                        VStack(alignment: .leading, spacing: 16) {
                            Text("Popular Recovery Topics")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            ForEach(insights.popularTopics) { topic in
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(topic.topic.capitalized)
                                            .font(.subheadline)
                                            .fontWeight(.medium)
                                        Text("\(topic.queryCount) queries")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    
                                    Spacer()
                                    
                                    VStack(alignment: .trailing) {
                                        HStack {
                                            ForEach(1...5, id: \.self) { star in
                                                Image(systemName: star <= Int(topic.avgRating) ? "star.fill" : "star")
                                                    .foregroundColor(.yellow)
                                                    .font(.caption)
                                            }
                                        }
                                        Text(String(format: "%.1f", topic.avgRating))
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(12)
                            }
                        }
                        
                        // User Engagement Section
                        VStack(alignment: .leading, spacing: 16) {
                            Text("Community Engagement")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            VStack(spacing: 12) {
                                MetricRow(
                                    title: "Daily Active Users",
                                    value: "\(insights.userEngagement.dailyActiveUsers)",
                                    icon: "person.3.fill"
                                )
                                
                                MetricRow(
                                    title: "Avg Session Duration",
                                    value: String(format: "%.1f min", insights.userEngagement.avgSessionDuration),
                                    icon: "clock.fill"
                                )
                                
                                MetricRow(
                                    title: "Completion Rate",
                                    value: String(format: "%.0f%%", insights.userEngagement.completionRate * 100),
                                    icon: "checkmark.circle.fill"
                                )
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                        }
                        
                        // Method Effectiveness Section
                        VStack(alignment: .leading, spacing: 16) {
                            Text("Method Effectiveness")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            VStack(spacing: 12) {
                                EffectivenessRow(
                                    title: "Top Rated Method",
                                    value: insights.methodEffectiveness.topRated,
                                    icon: "star.fill"
                                )
                                
                                EffectivenessRow(
                                    title: "Most Completed",
                                    value: insights.methodEffectiveness.mostCompleted,
                                    icon: "checkmark.seal.fill"
                                )
                                
                                EffectivenessRow(
                                    title: "Average Satisfaction",
                                    value: String(format: "%.1f/5.0", insights.methodEffectiveness.avgSatisfaction),
                                    icon: "heart.fill"
                                )
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                        }
                        
                    } else {
                        VStack(spacing: 16) {
                            ProgressView()
                                .scaleEffect(1.5)
                            Text("Loading insights...")
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 50)
                    }
                    
                    Spacer(minLength: 100)
                }
                .padding()
            }
            .navigationTitle("Analytics Insights")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
}

struct MetricRow: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.brandTeal)
                .frame(width: 20)
            
            Text(title)
                .font(.subheadline)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
    }
}

struct EffectivenessRow: View {
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.orange)
                .frame(width: 20)
            
            VStack(alignment: .leading) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            
            Spacer()
        }
    }
}
