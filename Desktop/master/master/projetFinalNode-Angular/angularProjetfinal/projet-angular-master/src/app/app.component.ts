import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Component, OnInit } from "@angular/core";
import { tap, switchMap } from "rxjs/operators";

@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrls: ["./app.component.css"]
})
export class AppComponent implements OnInit {
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };

  constructor(private http: HttpClient) {}

  ngOnInit() {
    
    // Step 1 : "Hello, Heroku ! 👋"
     this.http
      .post(
        "http://localhost:3000/api/login",
        { username: "lamine", password: "lamine" },
        this.httpOptions
      )

      .subscribe((res) => console.log(res));
    this.http
      .get("http://localhost:3000")
      .subscribe((res) => console.table(res));
      

    // Step 2 : "Get JWT token 🔓"
    this.http
      .post(
        "http://localhost:3000/api/login",
        { username: "lamine", password: "lamine" },
        this.httpOptions
      )
      // .pipe(
      //   tap((res) => console.log(res)),
      //   switchMap((res) => this.fetchPokemonlist(res.token))
      // )
      // .subscribe((res) => console.log(res));
      
  }

  // Step 3 : "Get pokemon list 🎉"
  fetchPokemonlist(token: string) {
    console.log("suis la");
    const httpOptionsWithJWT = {
      headers: new HttpHeaders({
        Authorization: `Bearer ${token}`
      })
    };
console.log("suis la")
    return this.http.get(
      "http://localhost:3000",
      httpOptionsWithJWT
    );
 }

}

