@app.route('/about', methods = ['GET'])
def about():
	return render_template('about.html')

@app.route('/contact', methods = ['GET'])
def contact():
	return render_template('contact.html')

@app.route('/get-a-qoute', methods = ['GET'])
def get_a_qoute():
	return render_template('get-a-qoute.html')

@app.route('/pricing', methods = ['GET'])
def pricing():
	return render_template('pricing.html')

@app.route('/sample-inner-page', methods = ['GET'])
def sample_inner_page():
	return render_template('sample-inner-page.html')

@app.route('/service-details', methods = ['GET'])
def service_details():
	return render_template('service-details.html')

@app.route('/services', methods = ['GET'])
def services():
	return render_template('services.html')
