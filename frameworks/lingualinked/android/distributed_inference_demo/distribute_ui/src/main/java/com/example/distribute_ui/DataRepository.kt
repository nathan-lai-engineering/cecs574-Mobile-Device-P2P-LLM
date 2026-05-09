package com.example.distribute_ui
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

object DataRepository {
    private val _isDirEmptyLiveData = MutableLiveData<Boolean>()
    val isDirEmptyLiveData: LiveData<Boolean> = _isDirEmptyLiveData

    private val _decodingStringLiveData = MutableLiveData<String>()
    val decodingStringLiveData: LiveData<String> = _decodingStringLiveData

    private val _sampleId = MutableLiveData<Int>()
    val sampleId: LiveData<Int> = _sampleId
    fun updateSampleId(sampleId: Int) {
        _sampleId.postValue(sampleId)
    }

    private val _ttft = MutableLiveData<Double>()
    val ttft: LiveData<Double> = _ttft
    fun updateTtft(ttft: Double) {
        _ttft.postValue(ttft)
    }

    private val _throughput = MutableLiveData<Double>()
    val throughput: LiveData<Double> = _throughput
    fun updateThroughput(throughput: Double) {
        _throughput.postValue(throughput)
    }

    private val _peakMemMb = MutableLiveData<Double>()
    val peakMemMb: LiveData<Double> = _peakMemMb
    fun updatePeakMem(mb: Double) {
        _peakMemMb.postValue(mb)
    }

    private val _peakBandwidthMbps = MutableLiveData<Double>()
    val peakBandwidthMbps: LiveData<Double> = _peakBandwidthMbps
    fun updatePeakBandwidth(mbps: Double) {
        _peakBandwidthMbps.postValue(mbps)
    }

    fun resetMetrics() {
        _ttft.postValue(null)
        _throughput.postValue(null)
        _peakMemMb.postValue(null)
        _peakBandwidthMbps.postValue(null)
    }

    fun updateDecodingString(updatedString: String) {
//        val responsePosition: Int = updatedString.indexOf("Response:")
//        val decodedStringAfterResponse: String = updatedString.substring(responsePosition + 9)
        _decodingStringLiveData.postValue(updatedString)
    }

    fun setIsDirEmpty(isEmpty: Boolean) {
        _isDirEmptyLiveData.postValue(isEmpty)
    }
}