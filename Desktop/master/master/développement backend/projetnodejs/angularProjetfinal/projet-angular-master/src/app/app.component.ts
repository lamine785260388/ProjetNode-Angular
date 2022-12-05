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
    // step 1Acces au serveur ! 👋"
    this.http
      .get("https://localhost:3000/")
      .subscribe((res) => console.log(res));

    // Step 2 : "Get JWT token 🔓"
    this.http
      .post(
        "https://localhost:3000/api/login",
        { username: "lamine", password: "lamine" },
        this.httpOptions
      )
   
  }

  // Step 3 : "Get pokemon list 🎉"
  fetchPokemonlist(token: string) {
    const httpOptionsWithJWT = {
      headers: new HttpHeaders({
        Authorization: `Bearer ${token}`
      })
    };

    return this.http.get(
      "https://localhost:3000/api/pokemons",
      httpOptionsWithJWT
    );
  }
}
