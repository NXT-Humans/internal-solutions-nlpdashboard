import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Clock, File, Activity, Loader } from 'lucide-react';

const ProcessingStatus = () => {
    const [processingStatus, setProcessingStatus] = useState({
        status: 'processing',
        steps: [],
        errors: [],
        timestamp: null,
        currentStage: '',
        processingComplete: false,
        documentCount: 0,
        processedFiles: {},
        overallProgress: 0
    });

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch('/api/upload-status');
                const data = await response.json();
                
                setProcessingStatus(prevStatus => ({
                    ...prevStatus,
                    ...data,
                    overallProgress: calculateProgress(data.steps)
                }));

                // Redirect to dashboard when processing is complete
                if (data.processing_complete) {
                    window.location.href = '/dashboard';
                }
            } catch (error) {
                console.error('Error fetching processing status:', error);
            }
        };

        // Initial fetch
        fetchStatus();

        // Poll for updates every 2 seconds
        const statusInterval = setInterval(fetchStatus, 2000);

        // Cleanup interval
        return () => clearInterval(statusInterval);
    }, []);

    const calculateProgress = (steps) => {
        if (!steps || steps.length === 0) return 0;
        
        const totalSteps = 6; // Expected total steps in pipeline
        return Math.min(Math.round((steps.length / totalSteps) * 100), 95);
    };

    const getStepIcon = (step) => {
        if (step.includes('error') || step.includes('failed')) {
            return <AlertTriangle className="h-5 w-5 text-red-500" />;
        }
        if (step.includes('completed') || step.includes('success')) {
            return <CheckCircle className="h-5 w-5 text-green-500" />;
        }
        if (step.includes('processing') || step.includes('analyzing')) {
            return <Activity className="h-5 w-5 text-blue-500 animate-pulse" />;
        }
        return <Clock className="h-5 w-5 text-gray-500" />;
    };

    const formatTimestamp = (timestamp) => {
        if (!timestamp) return '';
        return new Date(timestamp).toLocaleTimeString();
    };

    return (
        <div className="space-y-6">
            {/* Overall Progress */}
            <div className="bg-white rounded-lg shadow-sm p-6">
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-medium text-gray-900">
                            Processing Progress
                        </h3>
                        <span className="text-sm font-medium text-gray-500">
                            {processingStatus.overallProgress}%
                        </span>
                    </div>
                    
                    <div className="relative">
                        <div className="overflow-hidden h-2 flex rounded bg-gray-100">
                            <div
                                className={`transition-all duration-500 ease-out ${
                                    processingStatus.processingComplete
                                        ? 'bg-green-500'
                                        : 'bg-blue-500 animate-pulse'
                                }`}
                                style={{ width: `${processingStatus.overallProgress}%` }}
                            />
                        </div>
                    </div>

                    {processingStatus.currentStage && (
                        <p className="text-sm text-gray-600">
                            Currently: {processingStatus.currentStage}
                        </p>
                    )}
                </div>
            </div>

            {/* Processing Steps */}
            <div className="bg-white rounded-lg shadow-sm divide-y">
                <div className="p-4 bg-gray-50">
                    <h3 className="font-medium text-gray-900">Processing Steps</h3>
                </div>
                <div className="divide-y">
                    {processingStatus.steps.map((step, index) => (
                        <div
                            key={index}
                            className="flex items-start gap-3 p-4 hover:bg-gray-50 transition-colors"
                        >
                            {getStepIcon(step)}
                            <div className="min-w-0 flex-1">
                                <p className="text-sm font-medium text-gray-900">{step}</p>
                                <p className="text-sm text-gray-500">
                                    {formatTimestamp(processingStatus.timestamps?.[index])}
                                </p>
                            </div>
                        </div>
                    ))}

                    {processingStatus.steps.length === 0 && (
                        <div className="p-4 text-center text-gray-500">
                            <Loader className="h-5 w-5 animate-spin mx-auto mb-2" />
                            <p>Initializing processing...</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Errors Section */}
            {processingStatus.errors.length > 0 && (
                <div className="bg-red-50 rounded-lg shadow-sm overflow-hidden">
                    <div className="p-4 border-b border-red-100">
                        <h3 className="font-medium text-red-800">Processing Errors</h3>
                    </div>
                    <div className="divide-y divide-red-100">
                        {processingStatus.errors.map((error, index) => (
                            <div key={index} className="p-4 flex items-start gap-3">
                                <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0" />
                                <p className="text-sm text-red-700">{error}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Processing Complete Message */}
            {processingStatus.processingComplete && (
                <div className="bg-green-50 rounded-lg p-4 flex items-start gap-3">
                    <CheckCircle className="h-6 w-6 text-green-500 flex-shrink-0" />
                    <div>
                        <p className="font-medium text-green-800">Processing Complete</p>
                        <p className="text-sm text-green-700">
                            Redirecting to results dashboard...
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ProcessingStatus;