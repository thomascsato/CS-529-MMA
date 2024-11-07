import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private apiUrl = 'https://foi4xso7qi.execute-api.us-west-2.amazonaws.com/development/predict'; // Replace with your API URL

  constructor(private http: HttpClient) {}

  fetchData(inputData: any): Observable<any> {
    console.log('Sending data to Lambda:', inputData); // Log the input data

    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    })

    return this.http.post<any>(this.apiUrl, JSON.stringify(inputData), { headers });
  }
}