package httpserver

import (
	"context"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/kagent-dev/kagent/go/internal/a2a"
	"github.com/kagent-dev/kagent/go/internal/database"
	"github.com/kagent-dev/kagent/go/internal/httpserver/handlers"
	common "github.com/kagent-dev/kagent/go/internal/utils"
	"github.com/kagent-dev/kagent/go/internal/version"
	"github.com/kagent-dev/kagent/go/pkg/auth"
	"github.com/kagent-dev/kagent/go/pkg/client/api"
	"k8s.io/apimachinery/pkg/types"
	ctrl_client "sigs.k8s.io/controller-runtime/pkg/client"
	ctrllog "sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// API Path constants
	APIPathHealth          = "/health"
	APIPathVersion         = "/version"
	APIPathModelConfig     = "/modelconfigs"
	APIPathRuns            = "/runs"
	APIPathSessions        = "/sessions"
	APIPathTasks           = "/tasks"
	APIPathTools           = "/tools"
	APIPathToolServers     = "/toolservers"
	APIPathToolServerTypes = "/toolservertypes"
	APIPathAgents          = "/agents"
	APIPathProviders       = "/providers"
	APIPathModels          = "/models"
	APIPathMemories        = "/memories"
	APIPathNamespaces      = "/namespaces"
	APIPathA2A             = "/a2a"
	APIPathFeedback        = "/feedback"
	APIPathLangGraph       = "/langgraph"
	APIPathCrewAI          = "/crewai"
)

var defaultModelConfig = types.NamespacedName{
	Name:      "default-model-config",
	Namespace: common.GetResourceNamespace(),
}

// ServerConfig holds the configuration for the HTTP server
type ServerConfig struct {
	Router            *mux.Router
	BindAddr          string
	KubeClient        ctrl_client.Client
	A2AHandler        a2a.A2AHandlerMux
	WatchedNamespaces []string
	DbClient          database.Client
	Authenticator     auth.AuthProvider
	Authorizer        auth.Authorizer
}

// HTTPServer is the structure that manages the HTTP server
type HTTPServer struct {
	httpServer    *http.Server
	config        ServerConfig
	router        *mux.Router
	handlers      *handlers.Handlers
	dbManager     *database.Manager
	authenticator auth.AuthProvider
}

// NewHTTPServer creates a new HTTP server instance
func NewHTTPServer(config ServerConfig) (*HTTPServer, error) {
	// Initialize database

	return &HTTPServer{
		config:        config,
		router:        config.Router,
		handlers:      handlers.NewHandlers(config.KubeClient, defaultModelConfig, config.DbClient, config.WatchedNamespaces, config.Authorizer),
		authenticator: config.Authenticator,
	}, nil
}

// Start initializes and starts the HTTP server
func (s *HTTPServer) Start(ctx context.Context) error {
	log := ctrllog.FromContext(ctx).WithName("http-server")
	log.Info("Starting HTTP server", "address", s.config.BindAddr)

	// Setup routes
	s.setupRoutes()

	// Create HTTP server
	s.httpServer = &http.Server{
		Addr:    s.config.BindAddr,
		Handler: s.router,
	}

	// Start the server in a separate goroutine
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error(err, "HTTP server failed")
		}
	}()

	// Wait for context cancellation to shut down
	go func() {
		<-ctx.Done()
		log.Info("Shutting down HTTP server")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
			log.Error(err, "Failed to properly shutdown HTTP server")
		}
		// Close database connection
		if err := s.dbManager.Close(); err != nil {
			log.Error(err, "Failed to close database connection")
		}
	}()

	return nil
}

// Stop stops the HTTP server
func (s *HTTPServer) Stop(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// NeedLeaderElection implements controller-runtime's LeaderElectionRunnable interface
func (s *HTTPServer) NeedLeaderElection() bool {
	// Return false so the HTTP server runs on all instances, not just the leader
	return false
}

// setupRoutes configures all the routes for the server
func (s *HTTPServer) setupRoutes() {
	// Use middleware for common functionality (applies to all routes).
	s.router.Use(contentTypeMiddleware)
	s.router.Use(loggingMiddleware)
	s.router.Use(errorHandlerMiddleware)

	// Public routes (no authentication required).
	s.router.HandleFunc(APIPathHealth, adaptHealthHandler(s.handlers.Health.HandleHealth)).Methods(http.MethodGet)
	s.router.HandleFunc(APIPathVersion, adaptHandler(func(erw handlers.ErrorResponseWriter, r *http.Request) {
		versionResponse := api.VersionResponse{
			KAgentVersion: version.Version,
			GitCommit:     version.GitCommit,
			BuildDate:     version.BuildDate,
		}
		handlers.RespondWithJSON(erw, http.StatusOK, versionResponse)
	})).Methods(http.MethodGet)

	// Create a subrouter for authenticated API routes.
	apiRouter := s.router.PathPrefix("/api").Subrouter()
	apiRouter.Use(auth.AuthnMiddleware(s.authenticator))

	// Model configs
	apiRouter.HandleFunc(APIPathModelConfig, adaptHandler(s.handlers.ModelConfig.HandleListModelConfigs)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathModelConfig+"/{namespace}/{name}", adaptHandler(s.handlers.ModelConfig.HandleGetModelConfig)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathModelConfig, adaptHandler(s.handlers.ModelConfig.HandleCreateModelConfig)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathModelConfig+"/{namespace}/{name}", adaptHandler(s.handlers.ModelConfig.HandleDeleteModelConfig)).Methods(http.MethodDelete)
	apiRouter.HandleFunc(APIPathModelConfig+"/{namespace}/{name}", adaptHandler(s.handlers.ModelConfig.HandleUpdateModelConfig)).Methods(http.MethodPut)

	// Sessions - using database handlers
	apiRouter.HandleFunc(APIPathSessions, adaptHandler(s.handlers.Sessions.HandleListSessions)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathSessions, adaptHandler(s.handlers.Sessions.HandleCreateSession)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathSessions+"/agent/{namespace}/{name}", adaptHandler(s.handlers.Sessions.HandleGetSessionsForAgent)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathSessions+"/{session_id}", adaptHandler(s.handlers.Sessions.HandleGetSession)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathSessions+"/{session_id}/tasks", adaptHandler(s.handlers.Sessions.HandleListTasksForSession)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathSessions+"/{session_id}", adaptHandler(s.handlers.Sessions.HandleDeleteSession)).Methods(http.MethodDelete)
	apiRouter.HandleFunc(APIPathSessions+"/{session_id}", adaptHandler(s.handlers.Sessions.HandleUpdateSession)).Methods(http.MethodPut)
	apiRouter.HandleFunc(APIPathSessions+"/{session_id}/events", adaptHandler(s.handlers.Sessions.HandleAddEventToSession)).Methods(http.MethodPost)

	// Tasks
	apiRouter.HandleFunc(APIPathTasks+"/{task_id}", adaptHandler(s.handlers.Tasks.HandleGetTask)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathTasks, adaptHandler(s.handlers.Tasks.HandleCreateTask)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathTasks+"/{task_id}", adaptHandler(s.handlers.Tasks.HandleDeleteTask)).Methods(http.MethodDelete)

	// Tools - using database handlers
	apiRouter.HandleFunc(APIPathTools, adaptHandler(s.handlers.Tools.HandleListTools)).Methods(http.MethodGet)

	// Tool Servers
	apiRouter.HandleFunc(APIPathToolServers, adaptHandler(s.handlers.ToolServers.HandleListToolServers)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathToolServers, adaptHandler(s.handlers.ToolServers.HandleCreateToolServer)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathToolServers+"/{namespace}/{name}", adaptHandler(s.handlers.ToolServers.HandleDeleteToolServer)).Methods(http.MethodDelete)

	// Tool Server Types
	apiRouter.HandleFunc(APIPathToolServerTypes, adaptHandler(s.handlers.ToolServerTypes.HandleListToolServerTypes)).Methods(http.MethodGet)

	// Agents - using database handlers
	apiRouter.HandleFunc(APIPathAgents, adaptHandler(s.handlers.Agents.HandleListAgents)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathAgents, adaptHandler(s.handlers.Agents.HandleCreateAgent)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathAgents, adaptHandler(s.handlers.Agents.HandleUpdateAgent)).Methods(http.MethodPut)
	apiRouter.HandleFunc(APIPathAgents+"/{namespace}/{name}", adaptHandler(s.handlers.Agents.HandleGetAgent)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathAgents+"/{namespace}/{name}", adaptHandler(s.handlers.Agents.HandleDeleteAgent)).Methods(http.MethodDelete)

	// Providers
	apiRouter.HandleFunc(APIPathProviders+"/models", adaptHandler(s.handlers.Provider.HandleListSupportedModelProviders)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathProviders+"/memories", adaptHandler(s.handlers.Provider.HandleListSupportedMemoryProviders)).Methods(http.MethodGet)

	// Models
	apiRouter.HandleFunc(APIPathModels, adaptHandler(s.handlers.Model.HandleListSupportedModels)).Methods(http.MethodGet)

	// Memories
	apiRouter.HandleFunc(APIPathMemories, adaptHandler(s.handlers.Memory.HandleListMemories)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathMemories, adaptHandler(s.handlers.Memory.HandleCreateMemory)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathMemories+"/{namespace}/{name}", adaptHandler(s.handlers.Memory.HandleDeleteMemory)).Methods(http.MethodDelete)
	apiRouter.HandleFunc(APIPathMemories+"/{namespace}/{name}", adaptHandler(s.handlers.Memory.HandleGetMemory)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathMemories+"/{namespace}/{name}", adaptHandler(s.handlers.Memory.HandleUpdateMemory)).Methods(http.MethodPut)

	// Namespaces
	apiRouter.HandleFunc(APIPathNamespaces, adaptHandler(s.handlers.Namespaces.HandleListNamespaces)).Methods(http.MethodGet)

	// Feedback - using database handlers
	apiRouter.HandleFunc(APIPathFeedback, adaptHandler(s.handlers.Feedback.HandleCreateFeedback)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathFeedback, adaptHandler(s.handlers.Feedback.HandleListFeedback)).Methods(http.MethodGet)

	// LangGraph Checkpoints
	apiRouter.HandleFunc(APIPathLangGraph+"/checkpoints", adaptHandler(s.handlers.Checkpoints.HandlePutCheckpoint)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathLangGraph+"/checkpoints", adaptHandler(s.handlers.Checkpoints.HandleListCheckpoints)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathLangGraph+"/checkpoints/writes", adaptHandler(s.handlers.Checkpoints.HandlePutWrites)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathLangGraph+"/checkpoints/{thread_id}", adaptHandler(s.handlers.Checkpoints.HandleDeleteThread)).Methods(http.MethodDelete)

	// CrewAI
	apiRouter.HandleFunc(APIPathCrewAI+"/memory", adaptHandler(s.handlers.CrewAI.HandleStoreMemory)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathCrewAI+"/memory", adaptHandler(s.handlers.CrewAI.HandleGetMemory)).Methods(http.MethodGet)
	apiRouter.HandleFunc(APIPathCrewAI+"/memory", adaptHandler(s.handlers.CrewAI.HandleResetMemory)).Methods(http.MethodDelete)
	apiRouter.HandleFunc(APIPathCrewAI+"/flows/state", adaptHandler(s.handlers.CrewAI.HandleStoreFlowState)).Methods(http.MethodPost)
	apiRouter.HandleFunc(APIPathCrewAI+"/flows/state", adaptHandler(s.handlers.CrewAI.HandleGetFlowState)).Methods(http.MethodGet)

	// A2A
	apiRouter.PathPrefix(APIPathA2A + "/{namespace}/{name}").Handler(s.config.A2AHandler)
}

func adaptHandler(h func(handlers.ErrorResponseWriter, *http.Request)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h(w.(handlers.ErrorResponseWriter), r)
	}
}

func adaptHealthHandler(h func(http.ResponseWriter, *http.Request)) http.HandlerFunc {
	return h
}
